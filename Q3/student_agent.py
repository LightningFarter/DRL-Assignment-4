import gymnasium as gym
import numpy as np
from sac_icm_walk_stable import SACAgent, make_dmc_env


# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.env = make_dmc_env('humanoid-walk')
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.agent = SACAgent(self.state_dim, self.action_dim, self.action_space)
        self.agent.load_checkpoint('sac_icm_ckpt_walk_1300000.pth')


    def act(self, observation):
        return self.agent.select_action(observation)
