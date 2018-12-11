from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from math import exp, log
import math
import numpy as np

class EpsilonGreedyAgent():
    def __init__(self, world_shape, step_size, value_functions, epsilon, pos):
        self.world_shape = world_shape
        self.step_size = step_size
        self.value_functions = value_functions
        self.epsilon = epsilon
        self.pos = pos
        self.action_move_dict = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
            4: np.array([0, 1]),
        }

    def target_sample(self):
        action = 0
        cur_coord = _nodes_to_states(
            np.array([self.pos]),
            self.world_shape,
            self.step_size
        )[0]
        if np.random.random_sample() < self.epsilon:
            action = np.random.choice(5, 1)[0]
            new_coord = self.move_coordinate(cur_coord, action)
            self.pos = _states_to_nodes(
                np.array([new_coord]),
                self.world_shape,
                self.step_size
            )[0]
            return self.pos, action, new_coord

        best_value = -float('inf')
        best_next_state = cur_coord
        best_pos = self.pos
        best_action = 0
        for action in range(5):
            new_coord = self.move_coordinate(cur_coord, action)
            new_pos = _states_to_nodes(
                np.array([new_coord]),
                self.world_shape,
                self.step_size
            )[0]
            if self.value_functions[new_pos] > best_value:
                best_value = self.value_functions[new_pos]
                best_next_state = new_coord
                best_pos = new_pos
                best_action = action

        self.pos = best_pos
        return self.pos, action, best_next_state

    def move_coordinate(self, start_coord, action):
        new_coord = start_coord + self.action_move_dict[action] * self.step_size
        if new_coord[0] < 0.0:
            new_coord[0] = self.step_size[0] * (self.world_shape[0] - 1)
        if new_coord[0] > self.step_size[0] * (self.world_shape[0] - 1):
            new_coord[0] = 0.0
        if new_coord[1] < 0.0:
            new_coord[1] = self.step_size[1] * (self.world_shape[1] - 1)
        if new_coord[1] > self.step_size[1] * (self.world_shape[1] - 1):
            new_coord[1] = 0.0

        return new_coord

def _states_to_nodes(states, world_shape, step_size):
    """Convert physical states to node numbers.

    Parameters
    ----------
    states:      np.array
                 States with physical coordinates
    world_shape: tuple
                 The size of the grid_world
    step_size:   tuple
                 The step size of the grid world

    Returns
    -------
    nodes:       np.array
                 The node indices corresponding to the states
    """
    states = np.asanyarray(states)
    node_indices = np.rint(states / step_size).astype(np.int)
    return node_indices[:, 1] + world_shape[1] * node_indices[:, 0]

def _nodes_to_states(nodes, world_shape, step_size):
    """Convert node numbers to physical states.

    Parameters
    ----------
    nodes:          np.array
                    Node indices of the grid world
    world_shape:    tuple
                    The size of the grid_world
    step_size:      np.array
                    The step size of the grid world

    Returns
    -------
    states:         np.array
                    The states in physical coordinates
    """
    nodes = np.asanyarray(nodes)
    step_size = np.asanyarray(step_size)
    return np.vstack((nodes // world_shape[1],
                      nodes % world_shape[1])).T * step_size
