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
from collections import deque # Use deque for efficient buffer additions initially if preferred, but numpy array is better for sampling

# --- Hyperparameters ---
ENV_NAME = "humanoid-stand" # Changed to walk for a slightly more dynamic task than stand
SEED = 1
HIDDEN_DIM = 512 # Increased capacity slightly
ACTOR_LR = 3e-4 # Standard SAC LR, rely on clipping
CRITIC_LR = 3e-4 # Standard SAC LR, rely on clipping
ALPHA_LR = 3e-4 # Standard SAC LR
ICM_LR = 3e-4 # Standard SAC LR

GAMMA = 0.99         # Discount factor
TAU = 0.005        # Target network soft update rate
BUFFER_SIZE = int(1e6) # Larger buffer for complex tasks
BATCH_SIZE = 512
GRADIENT_STEPS = 1   # Number of gradient updates per environment step (after warmup)
LEARNING_STARTS = 10000 # Steps before starting gradient updates (fill buffer)
RANDOM_STEPS = 5000 # Initial steps with random actions for exploration

# --- ICM Specific ---
USE_ICM = True
ICM_ETA = 0.01 # Often needs tuning, start smaller
ICM_BETA = 0.2
ICM_FORWARD_LOSS_WEIGHT = 10.0 # Weight forward loss prediction more (common practice)
ICM_INVERSE_LOSS_WEIGHT = 0.1 # Weight inverse loss prediction less
INTRINSIC_REWARD_CLIP = 5.0 # Clip intrinsic reward magnitude
GRAD_CLIP_NORM = 5.0 # Max norm for gradient clipping

# --- Training & Saving ---
MAX_TIMESTEPS = int(3e6) # Train for a fixed number of timesteps instead of episodes
EVAL_FREQ = 50000     # Evaluate policy every N timesteps
SAVE_FREQ = 100000    # Save model every N timesteps
CHECKPOINT_DIR = 'checkpoints'
LOAD_MODEL_PATH = None # Set to path to resume training

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# Potentially speed up computations
# torch.backends.cudnn.benchmark = True


def make_dmc_env(env_name: str, seed=0, flatten=True) -> gym.Env:
    """Helper function to create dm_control environment"""
    domain_name, task_name = env_name.split('-')
    # Ensure task_kwargs={'random': seed} is correctly passed if needed by the specific task
    # Some tasks might not accept it directly in load. Check dm_control documentation.
    # Using environment's seed method is generally preferred.
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
    )
    env = DmControltoGymnasium(env, render_mode=None) # Add render_mode if needed
    # Important: Seed the environment action space for reproducibility
    env.action_space.seed(seed)
    if flatten and isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    return env

# --- ICM Module ---
class ICM(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=HIDDEN_DIM):
        super(ICM, self).__init__()
        # Feature extractor (optional but can be useful if input state is high-dim like pixels)
        # If state is already low-dim features, direct use is fine.
        # For simplicity, we'll omit a separate feature extractor here.

        # Inverse model: (s_feat, s'_feat) -> a_pred
        self.inverse_net = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), # LayerNorm can help stabilize
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Forward model: (s_feat, a) -> s'_feat_pred
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
        # In this version, features are the states themselves
        pred_action = self.inverse_net(torch.cat([state, next_state], dim=1))
        pred_next_state_feat = self.forward_net(torch.cat([state, action], dim=1))
        return pred_action, pred_next_state_feat

    def compute_intrinsic_reward(self, state: torch.Tensor, action: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # Use the learned forward model to predict the next state *feature*
            _, pred_next_state_feat = self.forward(state, action, next_state)
            # Intrinsic reward is the prediction error (MSE)
            # Use next_state directly if no separate feature extractor
            intrinsic_reward = F.mse_loss(pred_next_state_feat, next_state, reduction='none').mean(dim=1, keepdim=True)
        return intrinsic_reward

# --- Replay Buffer ---
class ReplayBufferNumpy:
    def __init__(self, state_dim: int, action_dim: int, max_size=BUFFER_SIZE):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

        self.device = device # Store device for direct tensor creation

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        # Store done as 0.0 or 1.0
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        ind = np.random.randint(0, self.size, size=batch_size)

        return {
            'states': torch.from_numpy(self.states[ind]).to(self.device),
            'actions': torch.from_numpy(self.actions[ind]).to(self.device),
            'rewards': torch.from_numpy(self.rewards[ind]).to(self.device),
            'next_states': torch.from_numpy(self.next_states[ind]).to(self.device),
            'dones': torch.from_numpy(self.dones[ind]).to(self.device)
        }

    def __len__(self) -> int:
        return self.size


# --- Networks ---
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=HIDDEN_DIM):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Optional: LayerNorm can sometimes help
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Optional
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Optional
            nn.ReLU(),
        )
        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        # Clamp log_std for stability
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        if deterministic:
            # Use mean for deterministic actions (e.g., evaluation)
            x_t = mean
        else:
            # Use rsample for reparameterization trick during training
            x_t = normal.rsample()

        # Apply Tanh squashing
        action = torch.tanh(x_t)

        # Calculate log probability with Tanh correction
        # log_prob = normal.log_prob(x_t)
        # log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6) # Epsilon for numerical stability
        # log_prob = log_prob.sum(dim=1, keepdim=True)

        # More numerically stable way from OpenAI's SpinningUp implementation
        log_prob = normal.log_prob(x_t).sum(axis=-1)
        log_prob -= (2 * (np.log(2) - x_t - F.softplus(-2 * x_t))).sum(axis=1)
        log_prob = log_prob.unsqueeze(-1) # Keep dimension consistent

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim=HIDDEN_DIM):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim + action_dim), # Input LayerNorm (less common)
            # nn.LayerNorm(hidden_dim), # Optional
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Optional
            nn.ReLU(),
             nn.Linear(hidden_dim, hidden_dim),
            # nn.LayerNorm(hidden_dim), # Optional
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        return self.net(x)

# --- SAC Agent ---
class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, action_space: gym.spaces.Space):
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1 = Critic(state_dim, action_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim).to(device)
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        # Freeze target networks
        for param in self.target_critic1.parameters(): param.requires_grad = False
        for param in self.target_critic2.parameters(): param.requires_grad = False

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic1_optim = optim.Adam(self.critic1.parameters(), lr=CRITIC_LR)
        self.critic2_optim = optim.Adam(self.critic2.parameters(), lr=CRITIC_LR)

        # Automatic entropy tuning
        self.target_entropy = -float(action_dim) # Target entropy is -|A|
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        self.alpha = self.log_alpha.exp().item() # Keep track of alpha value

        # ICM module
        self.use_icm = USE_ICM
        if self.use_icm:
            self.icm = ICM(state_dim, action_dim).to(device)
            self.icm_optim = optim.Adam(self.icm.parameters(), lr=ICM_LR)
            # Running mean/std for intrinsic reward normalization
            self.reward_rms = RunningMeanStd(shape=())


        self.gamma = GAMMA
        self.tau = TAU
        self.action_space = action_space # Store action space for clipping

        # Track losses
        self.actor_loss_val = 0
        self.critic1_loss_val = 0
        self.critic2_loss_val = 0
        self.alpha_loss_val = 0
        self.icm_loss_val = 0

    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action_tensor, _ = self.actor.sample(state_tensor, deterministic=deterministic)
            action = action_tensor.squeeze(0).cpu().numpy()
        # Clip action to ensure it's within the environment's bounds
        # Although Tanh squashes to [-1, 1], env might have different bounds
        return np.clip(action, self.action_space.low, self.action_space.high)

    def update(self, batch: Dict[str, torch.Tensor]) -> None:
        state = batch['states']
        action = batch['actions']
        extrinsic_reward = batch['rewards']
        next_state = batch['next_states']
        done = batch['dones']

        # --- ICM Update ---
        intrinsic_reward = torch.zeros_like(extrinsic_reward) # Default to zero if not used
        if self.use_icm:
            # Compute raw intrinsic reward (ensure ICM outputs float32)
            raw_intrinsic_reward = self.icm.compute_intrinsic_reward(state, action, next_state)

            # Normalize intrinsic reward using running mean/std
            self.reward_rms.update(raw_intrinsic_reward.cpu().numpy())

            # --- START FIX ---
            # Explicitly cast the variance tensor to float32
            variance = torch.tensor(self.reward_rms.var + 1e-8, dtype=torch.float32, device=device)
            norm_intrinsic_reward = raw_intrinsic_reward / torch.sqrt(variance)
            # --- END FIX ---

            # Clip normalized reward
            clipped_intrinsic_reward = torch.clamp(norm_intrinsic_reward, -INTRINSIC_REWARD_CLIP, INTRINSIC_REWARD_CLIP)

            intrinsic_reward = ICM_ETA * clipped_intrinsic_reward

            # Update ICM model
            # Ensure ICM inputs/outputs maintain float32
            pred_action, pred_next_state_feat = self.icm(state, action, next_state)
            forward_loss = F.mse_loss(pred_next_state_feat, next_state) # Assumes next_state is float32
            inverse_loss = F.mse_loss(pred_action, action) # Assumes action is float32

            icm_loss = (ICM_FORWARD_LOSS_WEIGHT * forward_loss +
                        ICM_INVERSE_LOSS_WEIGHT * inverse_loss)

            self.icm_optim.zero_grad()
            icm_loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(self.icm.parameters(), GRAD_CLIP_NORM)
            self.icm_optim.step()
            self.icm_loss_val = icm_loss.item()
        # --- End ICM Update ---

        # Combine rewards (ensure both are float32)
        # If raw_intrinsic_reward was float32 and variance tensor is now float32,
        # intrinsic_reward should also be float32.
        # extrinsic_reward comes from buffer, which should be float32.
        total_reward = extrinsic_reward + intrinsic_reward.detach()

        # --- Critic Update ---
        with torch.no_grad():
            # Ensure actor outputs float32
            next_action, next_log_prob = self.actor.sample(next_state)

            # Ensure target critics output float32
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q_min = torch.min(target_q1, target_q2) # Should be float32

            # Ensure alpha calculation maintains float32
            # self.alpha is python float, next_log_prob is float32 tensor -> result is float32
            target_q = target_q_min - self.alpha * next_log_prob

            # Compute TD target (if total_reward and target_q are float32, this should be float32)
            # done comes from buffer, should be float32
            td_target = total_reward + self.gamma * (1 - done) * target_q

        # Ensure critics output float32
        current_q1 = self.critic1(state, action)
        current_q2 = self.critic2(state, action)

        # Compute critic loss (MSE) - Now both inputs should be float32
        critic1_loss = F.mse_loss(current_q1, td_target)
        critic2_loss = F.mse_loss(current_q2, td_target)
        self.critic1_loss_val = critic1_loss.item()
        self.critic2_loss_val = critic2_loss.item()

        # Optimize Critic 1
        self.critic1_optim.zero_grad()
        critic1_loss.backward() # This was the failing line
        if GRAD_CLIP_NORM > 0:
             torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), GRAD_CLIP_NORM)
        self.critic1_optim.step()

        # Optimize Critic 2
        self.critic2_optim.zero_grad()
        critic2_loss.backward()
        if GRAD_CLIP_NORM > 0:
             torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), GRAD_CLIP_NORM)
        self.critic2_optim.step()

        # --- Actor Update ---
        # Sample new actions from current policy
        new_action, log_prob = self.actor.sample(state)

        # Compute Q values for the new actions
        q1_new = self.critic1(state, new_action)
        q2_new = self.critic2(state, new_action)
        q_min_new = torch.min(q1_new, q2_new)

        # Compute actor loss (policy gradient + entropy regularization)
        actor_loss = (self.alpha * log_prob - q_min_new).mean()
        self.actor_loss_val = actor_loss.item()

        # Optimize Actor
        self.actor_optim.zero_grad()
        actor_loss.backward()
        if GRAD_CLIP_NORM > 0:
             torch.nn.utils.clip_grad_norm_(self.actor.parameters(), GRAD_CLIP_NORM)
        self.actor_optim.step()

        # --- Alpha (Entropy Coefficient) Update ---
        # Use detached log_prob for alpha loss calculation
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_loss_val = alpha_loss.item()

        # Optimize Alpha
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        # No gradient clipping typically needed for single scalar alpha
        self.alpha_optim.step()
        self.alpha = self.log_alpha.exp().item() # Update alpha value

        # --- Target Network Soft Update ---
        self._soft_update(self.critic1, self.target_critic1)
        self._soft_update(self.critic2, self.target_critic2)

    def _soft_update(self, local_model: nn.Module, target_model: nn.Module):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_checkpoint(self, checkpoint_dir: str, timestep: int) -> None:
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f'sac_icm_ckpt_{timestep}.pth')
        checkpoint = {
            'timestep': timestep,
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic1_optim_state_dict': self.critic1_optim.state_dict(),
            'critic2_optim_state_dict': self.critic2_optim.state_dict(),
            'log_alpha': self.log_alpha.data, # Save data part of tensor
            'alpha_optim_state_dict': self.alpha_optim.state_dict(),
        }
        if self.use_icm:
            checkpoint.update({
                'icm_state_dict': self.icm.state_dict(),
                'icm_optim_state_dict': self.icm_optim.state_dict(),
                'reward_rms_mean': self.reward_rms.mean,
                'reward_rms_var': self.reward_rms.var,
                'reward_rms_count': self.reward_rms.count
            })
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> int:
        if not os.path.exists(checkpoint_path):
             print(f"Checkpoint file not found: {checkpoint_path}")
             return 0
        checkpoint = torch.load(checkpoint_path, map_location=device) # Ensure loading to correct device
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        # Load targets carefully
        self.target_critic1.load_state_dict(checkpoint.get('target_critic1_state_dict', self.critic1.state_dict())) # Fallback if missing
        self.target_critic2.load_state_dict(checkpoint.get('target_critic2_state_dict', self.critic2.state_dict())) # Fallback if missing

        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic1_optim.load_state_dict(checkpoint['critic1_optim_state_dict'])
        self.critic2_optim.load_state_dict(checkpoint['critic2_optim_state_dict'])

        # Load log_alpha correctly
        if 'log_alpha' in checkpoint:
             # Create a new tensor with requires_grad=True and copy data
             self.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=device)
             # Reload alpha optimizer with the new tensor
             self.alpha_optim = optim.Adam([self.log_alpha], lr=ALPHA_LR) # Recreate optimizer
             self.alpha_optim.load_state_dict(checkpoint['alpha_optim_state_dict']) # Load state
        self.alpha = self.log_alpha.exp().item()


        if self.use_icm and 'icm_state_dict' in checkpoint:
            self.icm.load_state_dict(checkpoint['icm_state_dict'])
            self.icm_optim.load_state_dict(checkpoint['icm_optim_state_dict'])
            # Load reward normalization stats
            self.reward_rms.mean = checkpoint.get('reward_rms_mean', np.zeros_like(self.reward_rms.mean))
            self.reward_rms.var = checkpoint.get('reward_rms_var', np.ones_like(self.reward_rms.var))
            self.reward_rms.count = checkpoint.get('reward_rms_count', 1e-4)


        start_timestep = checkpoint.get('timestep', 0) # Use get with default
        print(f"Loaded checkpoint from {checkpoint_path} at timestep {start_timestep}")
        return start_timestep

# Utility for Running Mean/Std normalization (from stable-baselines3)
class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = epsilon

    def update(self, arr: np.ndarray) -> None:
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean: np.ndarray, batch_var: np.ndarray, batch_count: int) -> None:
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m_2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count


# --- Evaluation Function ---
def evaluate_policy(env: gym.Env, agent: SACAgent, n_episodes: int = 5) -> float:
    total_reward = 0.0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action = agent.select_action(state, deterministic=True) # Use deterministic actions for eval
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / n_episodes

# --- Training Loop ---
def train(resume_checkpoint=LOAD_MODEL_PATH):
    print(f"Starting training on {ENV_NAME} with seed {SEED}")
    # Create training and evaluation environments
    env = make_dmc_env(ENV_NAME, seed=SEED)
    eval_env = make_dmc_env(ENV_NAME, seed=SEED + 100) # Use different seed for eval

    # Set random seeds
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED) # if use multi-GPU
    env.reset(seed=SEED) # Seed the environment reset

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_space = env.action_space

    agent = SACAgent(state_dim, action_dim, action_space)
    buffer = ReplayBufferNumpy(state_dim, action_dim, BUFFER_SIZE)

    start_timestep = 0
    if resume_checkpoint:
        start_timestep = agent.load_checkpoint(resume_checkpoint)

    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0
    evaluations = []

    start_time = time.time()

    print(f"Starting training loop from timestep {start_timestep}...")

    for t in range(start_timestep, MAX_TIMESTEPS):
        episode_timesteps += 1

        # Select action: random for initial steps, otherwise policy
        if t < RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store data in replay buffer
        # Use truncated flag to determine if the episode ended naturally or due to time limit
        # Store 1 if terminated, 0 if truncated or ongoing for 'done' signal in Bellman backup
        buffer.add(state, action, reward, next_state, float(terminated))

        state = next_state
        episode_reward += reward

        # Train agent after collecting enough data
        if t >= LEARNING_STARTS:
            for _ in range(GRADIENT_STEPS): # Perform multiple gradient steps per env step if desired
                 agent.update(buffer.sample(BATCH_SIZE))

        # If episode ends
        if done:
            episode_num += 1
            elapsed_time = time.time() - start_time
            print(f"T: {t+1} | Ep: {episode_num} | Ep Steps: {episode_timesteps} | Ep Reward: {episode_reward:.2f} | "
                  f"Buffer: {len(buffer)} | Alpha: {agent.alpha:.3f} | Time: {elapsed_time:.1f}s")

            # Log losses (optional, can be averaged over episode or logged periodically)
            # print(f"  Losses -> Actor: {agent.actor_loss_val:.3f}, Critic1: {agent.critic1_loss_val:.3f}, Critic2: {agent.critic2_loss_val:.3f}, Alpha: {agent.alpha_loss_val:.3f}, ICM: {agent.icm_loss_val:.3f}")


            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            start_time = time.time() # Reset timer for next episode info


        # Evaluate episode
        if (t + 1) % EVAL_FREQ == 0:
            eval_reward = evaluate_policy(eval_env, agent)
            evaluations.append(eval_reward)
            print("-" * 40)
            print(f"Evaluation over {5} episodes: {eval_reward:.3f} at timestep {t+1}")
            print("-" * 40)
            # Optionally save best model based on eval reward
            # np.save(f"./results/evaluations_{ENV_NAME}_{SEED}", evaluations)

        # Save checkpoint
        if (t + 1) % SAVE_FREQ == 0:
             agent.save_checkpoint(CHECKPOINT_DIR, t + 1)


    env.close()
    eval_env.close()
    print("Training finished.")


if __name__ == "__main__":
    train()