import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import os
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium
from typing import *
import time


HIDDEN_DIM = 256
OPTIM_LR = 3e-5
GAMMA = 0.99
TAU = 0.005

START_EPISODE = 0
MAX_EPISODE = 100000
BATCH_SIZE = 512
BUFFER_SIZE = 20000
SAVE_INTERVAL = 100

CHECKPOINT_DIR = 'checkpoints'
LOAD_MODEL_PATH = None

ENV = "humanoid-walk"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def make_dmc_env(env_name: str, seed=0, flatten=True) -> FlattenObservation:
    domain_name, task_name = env_name.split('-')
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={'random': seed}
    )
    env = DmControltoGymnasium(env)
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env

def reward_shaping(observation: np.ndarray) -> float:
    # Torso vertical orientation reward
    torso_vertical = observation['torso_vertical']
    target_torso_angle = np.array([0, 0, 1])  # Ideal vertical orientation
    torso_angle_diff = np.linalg.norm(torso_vertical - target_torso_angle)
    torso_reward = np.exp(-5 * torso_angle_diff)  # Gaussian-shaped reward
    torso_penalty = -0.2 * torso_angle_diff if torso_angle_diff > 0.15 else 0
    
    # Head height reward (normalized to humanoid size)
    current_head_height = observation['head_height'][0]
    target_head_height = 1.4  # Typical standing height :cite[4]
    head_height_diff = abs(current_head_height - target_head_height)
    head_reward = 1.0 - np.tanh(3 * head_height_diff)
    head_penalty = -0.1 * head_height_diff if head_height_diff > 0.2 else 0
    
    # COM velocity reward (forward motion focus)
    com_vel = observation['com_velocity']
    target_forward_vel = 1.2  # m/s (natural walking speed) :cite[2]
    vel_diff = abs(com_vel[0] - target_forward_vel)  # X-axis velocity
    velocity_reward = np.clip(com_vel[0], 0, target_forward_vel)/target_forward_vel
    velocity_penalty = -0.05 * vel_diff if vel_diff > 0.3 else 0
    
    # Joint angles smoothness bonus
    joint_angles = observation['joint_angles']
    joint_velocity = observation['velocity'][:21]
    smoothness_bonus = -0.01 * np.mean(np.abs(joint_velocity))
    
    # Combine components with safety prioritization :cite[3]
    total_reward = (
        0.4 * (torso_reward + torso_penalty) +  # Highest priority
        0.3 * (velocity_reward + velocity_penalty) +
        0.2 * (head_reward + head_penalty) +
        0.1 * smoothness_bonus
    )
    
    return np.clip(total_reward, -1, 1) * 100


class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE) -> None:
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0

    def add(self, state, action, reward, next_state, done) -> None:
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        indices = np.random.randint(0, len(self.buffer), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []

        for i in indices:
            s, a, r, s_, d = self.buffer[i]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(s_)
            dones.append(d)
        
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.FloatTensor(np.array(actions)).to(device),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, HIDDEN_DIM * 2)
        self.fc2 = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.mean = nn.Linear(HIDDEN_DIM // 2, action_dim)
        self.log_std = nn.Linear(HIDDEN_DIM // 2, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std

    def sample(self, state: torch.Tensor) -> Tuple[float, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, HIDDEN_DIM * 2)
        self.fc2 = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)
        self.fc3 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2)
        self.fc4 = nn.Linear(HIDDEN_DIM // 2, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int) -> None:
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=OPTIM_LR)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=OPTIM_LR)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=OPTIM_LR)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=OPTIM_LR)
        self.target_entropy = -action_dim

        self.gamma = GAMMA
        self.tau = TAU

    def update(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> None:
        state, action, reward, next_state, done = batch

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)

        self.critic1_optim.zero_grad()
        critic1_loss.backward()
        self.critic1_optim.step()

        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        self.critic2_optim.step()

        new_action, log_prob = self.actor.sample(state)
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        alpha = self.log_alpha.exp()

        q1 = self.critic1(state, new_action)
        q2 = self.critic2(state, new_action)
        actor_loss = (alpha * log_prob - torch.min(q1, q2)).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_checkpoint(self, checkpoint_dir: str, episode: int) -> None:
        checkpoint = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic1_optim_state_dict': self.critic1_optim.state_dict(),
            'critic2_optim_state_dict': self.critic2_optim.state_dict(),
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
            'log_alpha': self.log_alpha,
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, f'sac_walk_checkpoint_{episode}.pth'))
    
    def load_checkpoint(self, checkpoint_path) -> int:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim_state_dict'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])
        self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict'])
        self.log_alpha = checkpoint['log_alpha']

        return checkpoint['episode']


def train(resume_checkpoint=LOAD_MODEL_PATH):
    env = make_dmc_env(ENV)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer()

    print(ENV, state_dim, action_dim)

    start_episode = START_EPISODE
    max_episodes = MAX_EPISODE
    batch_size = BATCH_SIZE
    save_interval = SAVE_INTERVAL
    checkpoint_dir = CHECKPOINT_DIR

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    if resume_checkpoint:
        start_episode = agent.load_checkpoint(resume_checkpoint)
        print(f"Resuming training from episode {start_episode} using {resume_checkpoint}")

        buffer_path = resume_checkpoint.replace('.pth', '_buffer.npz')
        if os.path.exists(buffer_path):
            buffer_data = np.load(buffer_path, allow_pickle=True)
            buffer.buffer = list(buffer_data['buffer'])
            buffer.ptr = buffer_data['ptr']
            print(f"Loaded replay buffer with {len(buffer)} transitions")
    
    for episode in range(start_episode, max_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        episode_start_time = time.time()

        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action_tensor, _ = agent.actor.sample(state_tensor.unsqueeze(0))
                action = action_tensor.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                agent.update(batch)
        
        episode_time = time.time() - episode_start_time
        print(f"Episode {episode} | Reward: {episode_reward:.1f} | Time: {episode_time:.1f}s | Buffer: {len(buffer)}")

        if episode % save_interval == 0 or episode == max_episodes - 1:
            agent.save_checkpoint(checkpoint_dir, episode)
            buffer_path = os.path.join(checkpoint_dir, f'sac_walk_checkpoint_{episode}_buffer.npz')
            np.savez(buffer_path, buffer=np.array(buffer.buffer, dtype=object), ptr=buffer.ptr)
    
    env.close()


if __name__ == "__main__":
    train()