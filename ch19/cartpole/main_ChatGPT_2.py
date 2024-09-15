# coding: utf-8

import numpy as np
import gym
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import torch  # Import torch for GPU support

np.random.seed(1)

# Define a simple CartPole environment
class CustomCartPoleEnv(gym.Env):
    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()
        self.gravity = 9.8
        self.mass_cart = 1.0
        self.mass_pole = 0.1
        self.total_mass = self.mass_cart + self.mass_pole
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.mass_pole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.theta_threshold_radians = 12 * np.pi / 180
        self.x_threshold = 2.4

        # Define action and observation space
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=np.array([-self.x_threshold, -np.inf, -self.theta_threshold_radians, -np.inf]),
            high=np.array([self.x_threshold, np.inf, self.theta_threshold_radians, np.inf]),
            dtype=np.float32
        )
        self.state = None
        self.done = False
        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)  # Set seed for random library
        
    def reset(self):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.done = False
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state

        force = self.force_mag if action == 1 else -self.force_mag
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0 / 3.0 - self.mass_pole * costheta * costheta / self.total_mass))
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc

        self.state = np.array([x, x_dot, theta, theta_dot])
        
        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians

        reward = 1.0 if not done else 0.0

        return self.state, reward, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass

def plot_learning_history(total_rewards):
    plt.figure(figsize=(14, 5))
    plt.plot(total_rewards, lw=2, marker='o', markersize=5)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Total rewards', size=20)
    plt.title('Total rewards per episode')
    plt.show()

# General settings
EPISODES = 200

if __name__ == '__main__':
    env = DummyVecEnv([lambda: CustomCartPoleEnv()])
    
    # Check if GPU is available and set device accordingly
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using device: {device}')
    
    model = DQN('MlpPolicy', env, learning_rate=1e-4, buffer_size=50000, learning_starts=1000, batch_size=128, gamma=0.99, target_update_interval=100, exploration_fraction=0.2, exploration_final_eps=0.01, train_freq=4, gradient_steps=1, seed=1, verbose=1, device=device)  # Added device parameter

    total_rewards = []

    for episode in range(EPISODES):
        obs = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
        total_rewards.append(episode_reward)
        print(f'Episode: {episode + 1}/{EPISODES}, Total reward: {episode_reward}')

        # Train the model for 1000 timesteps after each episode
        model.learn(total_timesteps=1000, reset_num_timesteps=False)

    plot_learning_history(total_rewards)
