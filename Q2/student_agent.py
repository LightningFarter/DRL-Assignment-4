import gymnasium
import numpy as np
import torch
import torch.nn as nn

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

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gymnasium.spaces.Box(-1.0, 1.0, (1,), np.float64)
        state_dim = 5  # cart position, velocity, sin(theta), cos(theta)
        action_dim = 1
        self.policy_net = PolicyNetwork(state_dim, action_dim)
        self.policy_net.load_state_dict(torch.load("policy_checkpoint_5300224.pth"))
        self.policy_net.eval()

    def act(self, observation):
        state = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            mean, _ = self.policy_net(state)
            action = mean
        action_clipped = torch.clamp(action, -1.0, 1.0)
        return action_clipped.numpy()[0]
