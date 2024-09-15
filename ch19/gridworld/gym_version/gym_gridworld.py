# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 15:47:43 2024

@author: user
"""

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import gym
from gym import spaces

CELL_SIZE = 100
MARGIN = 10

def get_coords(row, col, loc='center'):
    xc = (col + 1.5) * CELL_SIZE
    yc = (row + 1.5) * CELL_SIZE
    if loc == 'center':
        return xc, yc
    elif loc == 'interior_corners':
        half_size = CELL_SIZE // 2 - MARGIN
        xl, xr = xc - half_size, xc + half_size
        yt, yb = yc - half_size, yc + half_size
        return [(xl, yt), (xr, yt), (xr, yb), (xl, yb)]
    elif loc == 'interior_triangle':
        x1, y1 = xc, yc + CELL_SIZE // 3
        x2, y2 = xc + CELL_SIZE // 3, yc - CELL_SIZE // 3
        x3, y3 = xc - CELL_SIZE // 3, yc - CELL_SIZE // 3
        return [(x1, y1), (x2, y2), (x3, y3)]

def draw_object(coords_list, ax):
    if len(coords_list) == 1:  # -> circle
        circle = plt.Circle(coords_list[0], radius=0.45 * CELL_SIZE, color='black')
        ax.add_artist(circle)
    elif len(coords_list) == 3:  # -> triangle
        triangle = plt.Polygon(coords_list, color='yellow')
        ax.add_artist(triangle)
    elif len(coords_list) > 3:  # -> polygon
        polygon = plt.Polygon(coords_list, color='blue')
        ax.add_artist(polygon)

class GridWorldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_rows=4, num_cols=6, delay=0.05):
        super(GridWorldEnv, self).__init__()
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.delay = delay

        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(num_cols * num_rows)

        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(self.observation_space.n)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(self.observation_space.n)}

        # Gold state
        gold_cell = (num_rows // 2, num_cols - 2)

        # Trap states
        trap_cells = [((gold_cell[0] + 1), gold_cell[1]),
                      (gold_cell[0], gold_cell[1] - 1),
                      ((gold_cell[0] - 1), gold_cell[1])]

        gold_state = self.grid2state_dict[gold_cell]
        trap_states = [self.grid2state_dict[(r, c)]
                       for (r, c) in trap_cells]
        self.terminal_states = [gold_state] + trap_states

        # Build the transition probability
        self.P = defaultdict(dict)
        for s in range(self.observation_space.n):
            row, col = self.state2grid_dict[s]
            self.P[s] = defaultdict(list)
            for a in range(self.action_space.n):
                action = self.action_defs[a]
                next_s = self.grid2state_dict[action(row, col)]

                # Terminal state
                if self.is_terminal(next_s):
                    r = (1.0 if next_s == self.terminal_states[0]
                         else -1.0)
                else:
                    r = 0.0
                if self.is_terminal(s):
                    done = True
                    next_s = s
                else:
                    done = False
                self.P[s][a] = [(1.0, next_s, r, done)]

        # Initial state distribution
        self.isd = np.zeros(self.observation_space.n)
        self.isd[0] = 1.0

        self.s = 0

    def is_terminal(self, state):
        return state in self.terminal_states

    def reset(self):
        self.s = np.random.choice(np.flatnonzero(self.isd))
        return self.s

    def step(self, action):
        row, col = self.state2grid_dict[self.s]
        action_func = self.action_defs[action]
        next_state = self.grid2state_dict[action_func(row, col)]
        prob, next_s, reward, done = self.P[self.s][action][0]
        self.s = next_s
        return next_s, reward, done, {}

    def render(self, mode='human', done=False):
        if done:
            sleep_time = 1
        else:
            sleep_time = self.delay

        fig, ax = plt.subplots(figsize=(self.num_cols, self.num_rows))
        ax.set_xlim(0, (self.num_cols + 2) * CELL_SIZE)
        ax.set_ylim(0, (self.num_rows + 2) * CELL_SIZE)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])

        # Draw grid
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                if (row, col) in self.terminal_states:
                    coords = get_coords(row, col, loc='interior_triangle')
                    draw_object(coords, ax)
                else:
                    coords = get_coords(row, col, loc='interior_corners')
                    draw_object(coords, ax)

        # Draw the agent
        agent_row, agent_col = self.state2grid_dict[self.s]
        agent_coords = get_coords(agent_row, agent_col, loc='center')
        draw_object([agent_coords], ax)

        plt.pause(sleep_time)
        plt.show()

    def close(self):
        plt.close()
