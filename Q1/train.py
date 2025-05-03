import gymnasium as gym
from ddpg_agent import DDPGAgent
import os

def train_agent(episodes=1000, save_interval=100, checkpoint_dir='checkpoints'):
    env = gym.make('Pendulum-v1')
    agent = DDPGAgent()
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # agent.load_checkpoint("checkpoints/checkpoint_1000.pth")
    # agent.noise = 0.0368

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        while True:
            action = agent.act(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.append((state, action, next_state, reward, done))
            agent.train_step()
            
            total_reward += reward
            state = next_state
            
            if done:
                break

        print(f'Episode {episode + 1} | Total Reward: {total_reward:.2f} | noise: {agent.noise:.4f}')

        if (episode + 1) % save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{episode + 1}.pth')
            agent.save_checkpoint(checkpoint_path)
            print(f'Saved checkpoint at {checkpoint_path}')
        
        agent.noise *= agent.noise_decay

if __name__ == '__main__':
    train_agent(episodes=1000)