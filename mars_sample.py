from __future__ import division, print_function, absolute_import

import time

import GPy
import matplotlib.pyplot as plt
import numpy as np

from multiagent_grid_world import (compute_true_safe_set, compute_S_hat0,
                                compute_true_S_hat, draw_gp_sample, MultiagentGridWorldAgent)
import pickle as pkl
from mars_utilities import mars_map

altitudes, coord, world_shape, step_size, num_of_points = mars_map()

# Define world
num_agent = 3
agent_explore_kernels = []
agent_explore_liks = []
agent_exploit_kernels = []
agent_exploit_liks = []
noise = 0.001

import copy
# Data to initialize GP
agent_gps = []
agent_rewards_gps = []
altitude_kernel = GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)
altitude_lik = GPy.likelihoods.Gaussian(variance=noise ** 2)
altitude_lik.constrain_bounded(1e-6, 10000.)
# print(altitudes)
# pkl.dump(altitudes, open('altitudes.pkl', 'wb'))

for agent in range(num_agent):
    agent_explore_kernels += [GPy.kern.RBF(input_dim=4, lengthscale=(2., 2., 2., 2.), ARD=True)]
    agent_explore_liks += [GPy.likelihoods.Gaussian(variance=noise ** 2)]
    agent_explore_liks[agent].constrain_bounded(1e-6, 1)

for agent in range(num_agent):
    agent_exploit_kernels += [GPy.kern.RBF(input_dim=4, lengthscale=(2., 2., 2., 2.), ARD=True)]
    agent_exploit_liks += [GPy.likelihoods.Gaussian(variance=noise ** 2)]
    agent_exploit_liks[agent].constrain_bounded(1e-6, 1)

agent_rewards_kernels = []
agent_rewards_liks = []
noise = 0.001

for agent in range(num_agent):
    agent_rewards_kernels += [GPy.kern.RBF(input_dim=2, lengthscale=(2., 2.), variance=1., ARD=True)]
    agent_rewards_liks += [GPy.likelihoods.Gaussian(variance=noise ** 2)]
    agent_rewards_liks[agent].constrain_bounded(1e-6, 10000)

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

for agent in range(num_agent):
    noise = 0.001
    n_samples = 1
    ind = np.random.choice(range(altitudes.size), n_samples)
    X = coord[ind, :]
    Y = altitudes[ind].reshape(n_samples, 1) + np.random.randn(n_samples, 1)
    agent_gps += [GPy.core.GP(X, Y, copy.deepcopy(altitude_kernel), copy.deepcopy(altitude_lik))]
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

    mean_func_exploit = GPy.core.Mapping(4, 1, name='agent_exploit')
    mean_func_exploit.f = lambda x: 0.5
    mean_func_exploit.update_gradients = lambda a,b: 0
    mean_func_exploit.gradients_X = lambda a,b: 0
    agent_exploit_gps += [GPy.core.GP(X, Y, agent_exploit_kernels[agent], agent_exploit_liks[agent], mean_function=mean_func_exploit)]

import copy
from tqdm import tqdm
collides = []
unsafes = []

for epoch in range(10):
    print(epoch)
    count_collide = 0
    count_unsafe = 0
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
        agents += [
            MultiagentGridWorldAgent(
                agent_gps[agent],
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

    for i in tqdm(range(10)):
        agent_actions = [0 for agent in range(num_agent)]
        agent_rewards = [0 for agent in range(num_agent)]
        agent_next_samples = []
        agent_states = []
        for agent in range(num_agent):
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
                count_unsafe += 1
    #         print()

    #     print(agent_states)
        if len(set(agent_states)) < len(agent_states):
            count_collide += 1
    #     print()

        for agent in range(num_agent):
            agents[agent].add_observation(agent_next_samples[agent][0], agent_next_samples[agent][1], agent_actions, agent_rewards)

    collides += [count_collide]
    unsafes += [count_unsafe]

print(collides)
print(unsafes)
print(np.mean(collides), np.std(collides))
print(np.mean(unsafes), np.std(unsafes))

import copy
from tqdm import tqdm
from singleagent_grid_world import SingleagentGridWorldAgent
collides = []
unsafes = []

for epoch in range(10):
    print(epoch)
    count_collide = 0
    count_unsafe = 0
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
                if not np.where(S_hat0[:, 1])[0][i] in agent_pos:
                    agent_pos += [np.where(S_hat0[:, 1])[0][i]]
                    break
            agent_S0s += [S0]
            agent_S_hat0s += [S_hat0]

    for agent in range(num_agent):
        cur_agent_explore_gps = copy.deepcopy(agent_explore_gps)
        cur_agent_explore_gps.pop(agent)
        cur_agent_exploit_gps = copy.deepcopy(agent_exploit_gps)
        cur_agent_exploit_gps.pop(agent)
        cur_agent_pos = copy.deepcopy(agent_pos)
        cur_agent_pos.pop(agent)
        agents += [
            SingleagentGridWorldAgent(
                agent_gps[agent],
                cur_agent_explore_gps,
                cur_agent_exploit_gps,
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

    for i in tqdm(range(10)):
        agent_actions = [0 for agent in range(num_agent)]
        agent_next_samples = []
        agent_states = []
        for agent in range(num_agent):
            agents[agent].update_sets()
            next_sample = agents[agent].target_sample()
    #         print('agent:' + str(agent))
    #         print(next_sample)
            agent_actions[agent] = next_sample[1]
            agent_next_samples += [next_sample]
            agent_states += [next_sample[0]]
            if altitudes[next_sample[0]] < h:
    #             print('entered an unsafe state')
                count_unsafe += 1
    #         print()

    #     print(agent_states)
        if len(set(agent_states)) < len(agent_states):
            count_collide += 1
    #     print()

        for agent in range(num_agent):
            agents[agent].add_observation(agent_next_samples[agent][0], agent_next_samples[agent][1], agent_actions)

    collides += [count_collide]
    unsafes += [count_unsafe]

print(collides)
print(unsafes)
print(np.mean(collides), np.std(collides))
print(np.mean(unsafes), np.std(unsafes))
