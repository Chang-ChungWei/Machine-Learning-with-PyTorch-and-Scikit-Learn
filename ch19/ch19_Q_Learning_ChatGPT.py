# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 23:03:03 2024

@author: user
"""

# coding: utf-8

# Python Machine Learning, PyTorch Edition by Sebastian Raschka (https://sebastianraschka.com), Yuxi (Hayden) Liu (https://www.mlexample.com/) & Vahid Mirjalili (http://vahidmirjalili.com), Packt Publishing Ltd. 2021
# Code Repository:
# Code License: MIT License (https://github.com/ LICENSE.txt)

#################################################################################
# Chapter 19 - Reinforcement Learning for Decision Making in Complex Environments
#################################################################################

# Script: gridworld_env.py

import numpy as np
from gym import Env
from gym.spaces import Discrete
from collections import defaultdict
import time
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Polygon

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

def draw_object(ax, coords_list):
    if len(coords_list) == 1:  # -> circle
        obj = plt.Circle(coords_list[0], 0.45 * CELL_SIZE, color='black')
    elif len(coords_list) == 3:  # -> triangle
        obj = Polygon(coords_list, closed=True, color='yellow')
    elif len(coords_list) > 3:  # -> polygon
        obj = Polygon(coords_list, closed=True, color='blue')
    return obj

class GridWorldEnv(Env):
    def __init__(self, num_rows=4, num_cols=6, delay=0.05):
        self.num_rows = num_rows
        self.num_cols = num_cols

        self.delay = delay

        move_up = lambda row, col: (max(row - 1, 0), col)
        move_down = lambda row, col: (min(row + 1, num_rows - 1), col)
        move_left = lambda row, col: (row, max(col - 1, 0))
        move_right = lambda row, col: (row, min(col + 1, num_cols - 1))

        self.action_defs = {0: move_up, 1: move_right,
                            2: move_down, 3: move_left}

        # Number of states/actions
        nS = num_cols * num_rows
        nA = len(self.action_defs)
        self.grid2state_dict = {(s // num_cols, s % num_cols): s
                                for s in range(nS)}
        self.state2grid_dict = {s: (s // num_cols, s % num_cols)
                                for s in range(nS)}

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
        P = defaultdict(dict)
        for s in range(nS):
            row, col = self.state2grid_dict[s]
            P[s] = defaultdict(list)
            for a in range(nA):
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
                P[s][a] = [(1.0, next_s, r, done)]

        # Initial state distribution
        isd = np.zeros(nS)
        isd[0] = 1.0

        self.nS = nS
        self.nA = nA
        self.P = P
        self.isd = isd

        self.viewer = None
        self._build_display(gold_cell, trap_cells)

    def is_terminal(self, state):
        return state in self.terminal_states

    def _build_display(self, gold_cell, trap_cells):
        self.fig, self.ax = plt.subplots(figsize=(self.num_cols, self.num_rows))
        self.ax.set_xlim(0, (self.num_cols + 2) * CELL_SIZE)
        self.ax.set_ylim(0, (self.num_rows + 2) * CELL_SIZE)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Draw grid
        for row in range(self.num_rows + 1):
            self.ax.plot([0, (self.num_cols + 1) * CELL_SIZE], [row * CELL_SIZE, row * CELL_SIZE], color='black')

        for col in range(self.num_cols + 1):
            self.ax.plot([col * CELL_SIZE, col * CELL_SIZE], [0, (self.num_rows + 1) * CELL_SIZE], color='black')

        # Traps
        for cell in trap_cells:
            trap_coords = get_coords(*cell, loc='center')
            self.ax.add_patch(draw_object(self.ax, [trap_coords]))

        # Gold
        gold_coords = get_coords(*gold_cell, loc='interior_triangle')
        self.ax.add_patch(draw_object(self.ax, gold_coords))

        # Agent
        if (os.path.exists('robot-coordinates.pkl') and CELL_SIZE == 100):
            agent_coords = pickle.load(open('robot-coordinates.pkl', 'rb'))
            starting_coords = get_coords(0, 0, loc='center')
            agent_coords += np.array(starting_coords)
        else:
            agent_coords = get_coords(0, 0, loc='interior_corners')
        self.agent_patch = draw_object(self.ax, agent_coords)
        self.ax.add_patch(self.agent_patch)

    def reset(self):
        self.s = 0
        return self.s

    def step(self, action):
        transitions = self.P[self.s][action]
        transition = transitions[0]  # deterministic environment
        next_state, reward, done = transition[1], transition[2], transition[3]
        self.s = next_state
        return next_state, reward, done

    def render(self, mode='human', done=False):
        if done:
            sleep_time = 1
        else:
            sleep_time = self.delay

        x_coord, y_coord = self.state2grid_dict[self.s]
        x_coord = (x_coord + 0.5) * CELL_SIZE
        y_coord = (y_coord + 0.5) * CELL_SIZE
        self.agent_patch.set_center((x_coord, y_coord))
        plt.draw()
        plt.pause(self.delay)

    def close(self):
        if self.viewer:
            plt.close(self.fig)
            self.viewer = None

if __name__ == '__main__':
    env = GridWorldEnv(5, 6)
    for i in range(1):
        s = env.reset()
        env.render(mode='human', done=False)

        while True:
            action = np.random.choice(env.nA)
            res = env.step(action)
            print('Action ', env.s, action, ' -> ', res)
            env.render(mode='human', done=res[2])
            if res[2]:
                break

    env.close()
