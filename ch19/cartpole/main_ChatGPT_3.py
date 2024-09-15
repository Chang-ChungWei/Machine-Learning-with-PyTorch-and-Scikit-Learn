import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from collections import namedtuple, deque
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy

np.random.seed(1)
torch.manual_seed(1)

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

        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.state = None
        self.done = False
        self.reset()

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
        self.done = done
        reward = 1.0 if not done else 0.0

        return self.state, reward, done, {}

    def render(self, mode='human'):
        # Rendering is omitted for simplicity
        pass

    def close(self):
        pass

def plot_learning_history(history):
    fig = plt.figure(1, figsize=(14, 5))
    ax = fig.add_subplot(1, 1, 1)
    episodes = np.arange(len(history)) + 1
    plt.plot(episodes, history, lw=4, marker='o', markersize=10)
    ax.tick_params(axis='both', which='major', labelsize=15)
    plt.xlabel('Episodes', size=20)
    plt.ylabel('Total rewards', size=20)
    plt.show()

if __name__ == '__main__':
    env = CustomCartPoleEnv()
    model = DQN('MlpPolicy', env, verbose=1, learning_rate=1e-3, buffer_size=2000, learning_starts=500, batch_size=32, gamma=0.95, train_freq=1, target_update_interval=10)
    model.learn(total_timesteps=10000)

    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f'Mean reward: {mean_reward} +/- {std_reward}')

    # Plot learning history (total rewards per episode)
    total_rewards = model.ep_info_buffer
    rewards = [ep_info['r'] for ep_info in total_rewards]
    plot_learning_history(rewards)
