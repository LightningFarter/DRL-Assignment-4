import gymnasium as gym
import numpy as np
import torch
from ddpg_agent import DDPGAgent


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.ddpg_agent = DDPGAgent()
        self.ddpg_agent.load_checkpoint('checkpoint_1000.pth')

    def act(self, observation):
        return self.ddpg_agent.act(observation, noise=False)


if __name__ == '__main__':
    env = gym.make('Pendulum-v1', render_mode='human')
    agent = Agent()

    state, _ = env.reset()
    total_reward = 0

    while True:
        action = agent.act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward

        if terminated or truncated:
            print(f"Test Total Reward: {total_reward}")
            break
