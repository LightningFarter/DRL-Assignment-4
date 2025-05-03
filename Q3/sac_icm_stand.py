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

USE_ICM = True
ICM_ETA = 0.1
ICM_BETA = 0.2
ICM_WARMUP_STEPS = 10000
RANDOM_ACTION_WARMUP = True
MIN_BUFFER_WARMUP = 5120
UPDATE_EVERY = 4

START_EPISODE = 0
MAX_EPISODE = 100000
BATCH_SIZE = 512
BUFFER_SIZE = 20000
SAVE_INTERVAL = 100
CHECKPOINT_DIR = 'checkpoints'
LOAD_MODEL_PATH = None
ENV = "humanoid-stand"

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


class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=HIDDEN_DIM):
        super(ICM, self).__init__()
        # Inverse model: (s, s') -> a
        self.inverse_net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Forward model: (s, a) -> s'
        self.forward_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pred_action = self.inverse_net(torch.cat([state, next_state], dim=1))
        pred_next_state = self.forward_net(torch.cat([state, action], dim=1))
        return pred_action, pred_next_state

    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, pred_next_state = self.forward(state, action, next_state)
            return F.mse_loss(pred_next_state, next_state, reduction='none').mean(dim=1, keepdim=True)


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
    def __init__(self, state_dim: int, action_dim: int):
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
        
        if USE_ICM:
            self.icm = ICM(state_dim, action_dim).to(device)
            self.icm_optim = optim.Adam(self.icm.parameters(), lr=OPTIM_LR)
            self.icm_trained_steps = 0
        
        self.gamma = GAMMA
        self.tau = TAU
    
    def update(self, batch: Tuple):
        state, action, extrinsic_reward, next_state, done = batch
        
        if USE_ICM:
            intrinsic_reward = ICM_ETA * self.icm.compute_intrinsic_reward(state, action, next_state)
            total_reward = extrinsic_reward + intrinsic_reward.detach()
            
            self.icm_optim.zero_grad()
            pred_action, pred_next_state = self.icm(state, action, next_state)
            inverse_loss = F.mse_loss(pred_action, action)
            forward_loss = F.mse_loss(pred_next_state, next_state)
            (inverse_loss + ICM_BETA * forward_loss).backward()
            self.icm_optim.step()
        else:
            total_reward = extrinsic_reward
        
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q = torch.min(self.target_critic1(next_state, next_action), self.target_critic2(next_state, next_action)) - self.log_alpha.exp() * next_log_prob
            target_q = total_reward + (1 - done) * self.gamma * target_q
        
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
        
        q1, q2 = self.critic1(state, new_action), self.critic2(state, new_action)
        actor_loss = (self.log_alpha.exp() * log_prob - torch.min(q1, q2)).mean()
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
        if USE_ICM:
            checkpoint.update({
                'icm_state_dict': self.icm.state_dict(),
                'icm_optim_state_dict': self.icm_optim.state_dict()
            })
        torch.save(checkpoint, os.path.join(checkpoint_dir, f'sac_icm_checkpoint_{episode}.pth'))
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path)
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
        if USE_ICM and 'icm_state_dict' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
            self.icm_optim.load_state_dict(checkpoint['icm_optim_state_dict'])
        return checkpoint['episode']

    def pretrain_icm(self, batch: Tuple) -> float:
        """Special training step for ICM warm-up"""
        state, action, _, next_state, _ = batch
        self.icm_optim.zero_grad()
        pred_action, pred_next_state = self.icm(state, action, next_state)
        
        # Calculate losses with layer-wise normalization
        inverse_loss = F.mse_loss(pred_action, action)
        forward_loss = F.mse_loss(pred_next_state, next_state)
        total_loss = inverse_loss + ICM_BETA * forward_loss
        
        total_loss.backward()
        self.icm_optim.step()
        self.icm_trained_steps += 1
        return total_loss.item()


def train(resume_checkpoint=LOAD_MODEL_PATH):
    env = make_dmc_env(ENV)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = SACAgent(state_dim, action_dim)
    buffer = ReplayBuffer()
    
    start_episode = START_EPISODE
    if resume_checkpoint:
        start_episode = agent.load_checkpoint(resume_checkpoint)
        print(f"Resumed from checkpoint {resume_checkpoint}")
    
    total_steps = 0
    in_warmup = True
    
    for episode in range(start_episode, MAX_EPISODE):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        start_time = time.time()
        
        while not done:
            # Action selection logic with warmup support
            if in_warmup and RANDOM_ACTION_WARMUP:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).to(device)
                    action_tensor, _ = agent.actor.sample(state_tensor.unsqueeze(0))
                    action = action_tensor.squeeze(0).cpu().numpy()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            total_steps += 1

            # ICM warm-up phase logic
            if in_warmup:
                
                if len(buffer) >= MIN_BUFFER_WARMUP and total_steps % UPDATE_EVERY == 0:
                    warmup_batch = buffer.sample(min(256, len(buffer)))
                    icm_loss = agent.pretrain_icm(warmup_batch)
                    
                    # Print warmup progress every 10%
                    if agent.icm_trained_steps % (ICM_WARMUP_STEPS//10) == 0:
                        print(f"ICM Warmup: {agent.icm_trained_steps/ICM_WARMUP_STEPS*100:.0f}% "
                              f"| Loss: {icm_loss:.3f}")
                
                # Check warmup completion
                if agent.icm_trained_steps >= ICM_WARMUP_STEPS and total_steps % UPDATE_EVERY == 0:
                    print("\nICM Warmup Complete!")
                    in_warmup = False

            else:
                # Normal training phase
                if len(buffer) >= BATCH_SIZE:
                    agent.update(buffer.sample(BATCH_SIZE))

        # Post-episode logging
        time_used = time.time() - start_time

        phase = "Warmup" if in_warmup else "Training"
        print(f"[{phase}] Episode {episode} | Reward: {episode_reward:.1f} | "
              f"Total Steps: {total_steps} | Buffer: {len(buffer)} | Time: {time_used:.1f}")
        
        # Checkpoint saving (only after warmup)
        if not in_warmup and episode % SAVE_INTERVAL == 0:
            agent.save_checkpoint(CHECKPOINT_DIR, episode)
    
    env.close()


if __name__ == "__main__":
    train()