from __future__ import division, print_function, absolute_import

import time

import GPy
import matplotlib.pyplot as plt
import numpy as np

from multiagent_grid_world import (compute_true_safe_set, compute_S_hat0,
                                compute_true_S_hat, draw_gp_sample, MultiagentGridWorldAgent)
from epsilon_greedy_agent import EpsilonGreedyAgent
import pickle as pkl

from mars_utilities import mars_map

altitudes, coord, world_shape, step_size, num_of_points = mars_map()

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

num_agent = 10
agent_explore_kernels = []
agent_explore_liks = []
noise = 0.001

import copy
# Data to initialize GP
agent_gps = []
agent_rewards_gps = []
altitude_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
altitude_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
altitude_lik.constrain_bounded(1e-6, 10000.)

for agent in range(num_agent):
    agent_explore_kernels += [GPy.kern.RBF(input_dim=4, lengthscale=(2., 2., 2., 2.), ARD=True)]
    agent_explore_liks += [GPy.likelihoods.Gaussian(variance=noise ** 2)]
    agent_explore_liks[agent].constrain_bounded(1e-6, 1)

# Define coordinates
n, m = world_shape
step1, step2 = step_size
xx, yy = np.meshgrid(np.linspace(0, (n - 1) * step1, n),
                     np.linspace(0, (m - 1) * step2, m),
                     indexing="ij")
coord = np.vstack((xx.flatten(), yy.flatten())).T

# Safety threhsold
h = -0.5

# Lipschitz
L = 0

# Scaling factor for confidence interval
beta = 2

collide_threshold = 0.95

ind = np.random.choice(range(altitudes.size), 1)
X = coord[ind, :]
Y = altitudes[ind].reshape(1, 1) + np.random.randn(1, 1)
agent_gp = GPy.core.GP(X, Y, copy.deepcopy(altitude_kernel), copy.deepcopy(altitude_lik))
for agent in range(num_agent):
    noise = 0.001
    n_samples = 1
    ind = np.random.choice(range(altitudes.size), n_samples)
    X = coord[ind, :]
    Y = altitudes[ind].reshape(n_samples, 1) + np.random.randn(n_samples, 1)
    agent_rewards_gps += [GPy.core.GP(X, Y, copy.deepcopy(altitude_kernel), copy.deepcopy(altitude_lik))]

import numpy as np

n_samples = 1
agent_explore_gps = []
agent_exploit_gps = []
X = []
Y = []

def move_coordinate(start_coord, action):
    action_move_dict = {
        0: np.array([0, 0]),
        1: np.array([-1, 0]),
        2: np.array([0, -1]),
        3: np.array([1, 0]),
        4: np.array([0, 1]),
    }
    new_coord = start_coord + action_move_dict[action] * step_size
    if new_coord[0] < 0.0:
        new_coord[0] = step_size[0] * (world_shape[0] - 1)
    if new_coord[0] > step_size[0] * (world_shape[0] - 1):
        new_coord[0] = 0.0
    if new_coord[1] < 0.0:
        new_coord[1] = step_size[1] * (world_shape[1] - 1)
    if new_coord[1] > step_size[1] * (world_shape[1] - 1):
        new_coord[1] = 0.0

    return new_coord

for agent in range(num_agent):
    for i in range(world_shape[0]):
        for j in range(world_shape[1]):
            for action in range(0, 5):
                current_state = np.array([i * step_size[0], j * step_size[1]])
                next_state = move_coordinate(current_state, action)
                X += [[i * step_size[0], j * step_size[1], next_state[0], next_state[1]]]
                Y += [[np.random.uniform(-1,0,1000)[0]]]

    X = np.array(X)
    Y = np.array(Y)
    mean_func_explore = GPy.core.Mapping(4, 1, name='agent_explore')
    mean_func_explore.f = lambda x: 0.5
    mean_func_explore.update_gradients = lambda a,b: 0
    mean_func_explore.gradients_X = lambda a,b: 0
    agent_explore_gps += [GPy.core.GP(X, Y, agent_explore_kernels[agent], agent_explore_liks[agent], mean_function=mean_func_explore)]

    difference = 10000

    value_functions = [0.0 for i in range(world_shape[0] * world_shape[1])]
    while difference > 0.01:
        cur_difference = 0.0
        num_nodes = world_shape[0] * world_shape[1]
        for node in range(num_nodes):
            state = _nodes_to_states(
                np.array([node]),
                world_shape,
                step_size,
            )[0]
            cur_reward = altitudes[node]
            old_value = value_functions[node]
            for action in range(0, 5):
                new_state = move_coordinate(state, action)
                new_node = _states_to_nodes(
                    np.array([new_state]),
                    world_shape,
                    step_size
                )[0]
                value_functions[node] = max(
                    cur_reward + 0.9 * value_functions[new_node],
                    old_value
                )

            cur_difference = max(
                cur_difference,
                abs(value_functions[node] - old_value)
            )

        difference = cur_difference

import copy
from tqdm import tqdm
algorithm_collides = []
other_collides = []
algorithm_unsafes = []
other_unsafes = []
algorithm_cumulative_altitudes = []
other_cumulative_altitudes = []

for epoch in range(10):
    print(epoch)
    count_algorithm_collide = 0
    count_other_collide = 0
    count_algorithm_unsafe = 0
    count_other_unsafe = 0
    algorithm_cumulative_altitude = 0
    other_cumulative_altitude = 0
    agents = []
    agent_S0s = []
    agent_S_hat0s = []
    agent_pos = []
    while len(agent_pos) != num_agent:
        agent_pos = []
        for agent in range(num_agent):
            S0 = np.zeros((np.prod(world_shape), 5), dtype=bool)
            S0[:, 0] = True
            S_hat0 = compute_S_hat0(np.nan, world_shape, 4, altitudes,
                                    step_size, h)
            for i in range(len(np.where(S_hat0[:, 1])[0])):
                if np.where(S_hat0[:, 1])[0][i] not in agent_pos:
                    agent_pos += [np.where(S_hat0[:, 1])[0][i]]
                    break
            agent_S0s += [S0]
            agent_S_hat0s += [S_hat0]

    for agent in range(num_agent):
        cur_agent_explore_gps = copy.deepcopy(agent_explore_gps)
        cur_agent_explore_gps.pop(agent)
        cur_agent_rewards_gps = copy.deepcopy(agent_rewards_gps)
        cur_agent_rewards_gps.pop(agent)
        cur_agent_pos = copy.deepcopy(agent_pos)
        cur_agent_pos.pop(agent)
        if agent == 0:
            agents += [
                MultiagentGridWorldAgent(
                    agent_gp,
                    cur_agent_explore_gps,
                    cur_agent_rewards_gps,
                    world_shape,
                    step_size,
                    beta,
                    altitudes,
                    h,
                    collide_threshold,
                    agent_S0s[agent],
                    agent_S_hat0s[agent],
                    agent_pos[agent],
                    L,
                    cur_agent_pos,
                    [0.5 for _ in range(num_agent - 1)],
                )
            ]
        else:
            agents += [
                EpsilonGreedyAgent(
                    world_shape,
                    step_size,
                    value_functions,
                    0.9,
                    agent_pos[agent]
                )
            ]

    for i in tqdm(range(10)):
        agent_actions = [0 for agent in range(num_agent)]
        agent_rewards = [0 for agent in range(num_agent)]
        agent_next_samples = []
        agent_states = []
        for agent in range(num_agent):
            if agent == 0:
                agents[agent].update_sets()
            next_sample = agents[agent].target_sample()
    #         print('agent:' + str(agent))
    #         print(next_sample)
            agent_actions[agent] = next_sample[1]
            agent_rewards[agent] = altitudes[next_sample[0]]
            agent_next_samples += [next_sample]
            agent_states += [next_sample[0]]
            if altitudes[next_sample[0]] < h:
    #             print('entered an unsafe state')
                if agent == 0:
                    count_algorithm_unsafe += 1
                else:
                    count_other_unsafe += 1
    #         print()
            if agent == 0:
                algorithm_cumulative_altitude += altitudes[next_sample[0]]
            else:
                other_cumulative_altitude += altitudes[next_sample[0]]

    #     print(agent_states)
        if agent_states[0] in agent_states[1:]:
            count_algorithm_collide += 1
        for agent in range(1, num_agent):
            if agent_states[agent] in (agent_states[:agent] + agent_states[agent + 1:]):
                count_other_collide += 1
    #     print()

        agents[0].add_observation(agent_next_samples[agent][0], agent_next_samples[agent][1], agent_actions, agent_rewards)

    algorithm_collides += [count_algorithm_collide]
    algorithm_unsafes += [count_algorithm_unsafe]
    other_collides += [count_other_collide]
    other_unsafes += [count_other_unsafe]
    algorithm_cumulative_altitudes += [algorithm_cumulative_altitude]
    other_cumulative_altitudes += [other_cumulative_altitude]

print(algorithm_collides)
print(algorithm_unsafes)
print(other_collides)
print(other_unsafes)
print(np.mean(algorithm_collides), np.std(algorithm_collides))
print(np.mean(algorithm_unsafes), np.std(algorithm_unsafes))
print(np.mean(other_collides), np.std(other_collides))
print(np.mean(other_unsafes), np.std(other_unsafes))
print(np.mean(algorithm_cumulative_altitudes), np.std(algorithm_cumulative_altitudes))
print(np.mean(other_cumulative_altitudes), np.std(other_cumulative_altitudes))
