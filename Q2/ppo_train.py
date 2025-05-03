import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium

def make_dmc_env(env_name, seed, flatten=True, use_pixels=False):
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )
    env = DmControltoGymnasium(env, render_mode="rgb_array", render_kwargs={"width": 256, "height": 256, "camera_id": 0})
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        return mean, self.log_std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.value = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        value = self.value(x)
        return value


class PPOTrainer:
    def __init__(self, env, policy_net, value_net, policy_optimizer, value_optimizer):
        self.env = env
        self.policy_net = policy_net
        self.value_net = value_net
        self.policy_optimizer = policy_optimizer
        self.value_optimizer = value_optimizer
        self.gamma = 0.99
        self.lambda_ = 0.95
        self.eps = 0.2
        self.c1 = 0.5  # value loss coefficient
        self.c2 = 0.01  # entropy coefficient
        self.epochs = 10
        self.batch_size = 64
        self.max_timesteps = 10000000

    def collect_trajectories(self, num_steps):
        states, actions, rewards, dones, log_probs, values, next_states = [], [], [], [], [], [], []
        state = self.env.reset()
        if isinstance(state, tuple):  # Handle case where reset returns (state, info)
            state = state[0]
        steps = 0
        while steps < num_steps:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                mean, log_std = self.policy_net(state_tensor)
                action_dist = Normal(mean, log_std.exp())
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action).sum(dim=-1)
                value = self.value_net(state_tensor)
            action_clipped = torch.clamp(action, -1.0, 1.0)
            next_state, reward, done, truncated, _ = self.env.step(action_clipped.numpy()[0])
            done = done or truncated  # Consider truncated as done
            if isinstance(next_state, tuple):  # Handle case where step returns tuple
                next_state = next_state[0]
            states.append(state)
            actions.append(action.numpy()[0])
            rewards.append(reward)
            dones.append(done)
            log_probs.append(log_prob.numpy()[0])
            values.append(value.numpy()[0][0])
            next_states.append(next_state)
            state = next_state
            steps += 1
            if done:
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]
        return states, actions, rewards, dones, log_probs, values, next_states

    def compute_gae(self, rewards, dones, values, next_states):
        advantages = []
        advantage = 0
        next_values = []
        for i in range(len(next_states)):
            if dones[i]:
                next_values.append(0)
            else:
                next_state_tensor = torch.FloatTensor(next_states[i]).unsqueeze(0)
                with torch.no_grad():
                    next_value = self.value_net(next_state_tensor).item()
                next_values.append(next_value)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] - values[t]
            advantage = delta + self.gamma * self.lambda_ * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return advantages

    def train(self):
        total_timesteps = 0
        episode_rewards = []
        while total_timesteps < self.max_timesteps:
            states, actions, rewards, dones, log_probs, values, next_states = self.collect_trajectories(2048)
            total_timesteps += len(states)
            ep_reward = sum(rewards)
            episode_rewards.append(ep_reward)
            advantages = self.compute_gae(rewards, dones, values, next_states)
            returns = [adv + val for adv, val in zip(advantages, values)]
            states_tensor = torch.FloatTensor(states)
            actions_tensor = torch.FloatTensor(actions)
            log_probs_tensor = torch.FloatTensor(log_probs)
            advantages_tensor = torch.FloatTensor(advantages)
            returns_tensor = torch.FloatTensor(returns)
            advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
            for _ in range(self.epochs):
                indices = np.random.choice(len(states), self.batch_size, replace=False)
                batch_states = states_tensor[indices]
                batch_actions = actions_tensor[indices]
                batch_log_probs = log_probs_tensor[indices]
                batch_advantages = advantages_tensor[indices]
                batch_returns = returns_tensor[indices]
                mean, log_std = self.policy_net(batch_states)
                action_dist = Normal(mean, log_std.exp())
                current_log_probs = action_dist.log_prob(batch_actions).sum(dim=-1)
                entropy = action_dist.entropy().mean()
                values = self.value_net(batch_states).squeeze()
                ratio = torch.exp(current_log_probs - batch_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (batch_returns - values).pow(2).mean()
                policy_total_loss = policy_loss - self.c2 * entropy
                self.policy_optimizer.zero_grad()
                policy_total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()
            if total_timesteps % 100000 < 2048:  # Save approximately every 50000 steps
                torch.save(self.policy_net.state_dict(), f"checkpoints/policy_checkpoint_{total_timesteps}.pth")
                print(f"Checkpoint saved at timestep {total_timesteps}, Avg Reward: {np.mean(episode_rewards[-10:]):.2f}")
        torch.save(self.policy_net.state_dict(), "policy_final.pth")
        print(f"Training completed. Final average reward: {np.mean(episode_rewards[-10:]):.2f}")

def train_cartpole():
    env = make_dmc_env("cartpole-balance", seed=np.random.randint(0, 1000000), flatten=True, use_pixels=False)
    state_dim = env.observation_space.shape[0]  # Should be 4: cart pos, vel, sin(theta), cos(theta)
    action_dim = env.action_space.shape[0]     # 1: force applied to cart
    policy_net = PolicyNetwork(state_dim, action_dim)
    value_net = ValueNetwork(state_dim)
    policy_optimizer = optim.Adam(policy_net.parameters(), lr=3e-4)
    value_optimizer = optim.Adam(value_net.parameters(), lr=3e-4)
    trainer = PPOTrainer(env, policy_net, value_net, policy_optimizer, value_optimizer)
    trainer.train()
    env.close()

if __name__ == "__main__":
    train_cartpole()