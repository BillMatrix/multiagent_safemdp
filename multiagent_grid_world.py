from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

from utilities import DifferenceKernel
from SafeMDP_class import (reachable_set, returnable_set, SafeMDP,
                            link_graph_and_safe_set)
from math import exp, log
import math

def compute_true_safe_set(world_shape, altitude, h):
    """
    Computes the safe set given a perfect knowledge of the map

    Parameters
    ----------
    world_shape: tuple
    altitude: np.array
        1-d vector with altitudes for each node
    h: float
        Safety threshold for height differences

    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """

    true_safe = np.zeros((world_shape[0] * world_shape[1], 5), dtype=np.bool)

    altitude_grid = altitude.reshape(world_shape)

    # Reshape so that first dimensions are actions, the rest is the grid world.
    safe_grid = true_safe.T.reshape((5,) + world_shape)

    # Height difference (next height - current height) --> positive if downhill
    up_diff = altitude_grid[:, :-1] - altitude_grid[:, 1:]
    right_diff = altitude_grid[:-1, :] - altitude_grid[1:, :]

    # State are always safe
    true_safe[:, 0] = True

    # Going in the opposite direction
    safe_grid[1, :, :-1] = up_diff >= h
    safe_grid[2, :-1, :] = right_diff >= h
    safe_grid[3, :, 1:] = -up_diff >= h
    safe_grid[4, 1:, :] = -right_diff >= h

    return true_safe

def compute_S_hat0(s, world_shape, n_actions, altitudes, step_size, h):
    """
    Compute a valid initial safe seed.

    Parameters
    ---------
    s: int or nan
        Vector index of the state where we start computing the safe seed
        from. If it is equal to nan, a state is chosen at random
    world_shape: tuple
        Size of the grid world (rows, columns)
    n_actions: int
        Number of actions available to the agent
    altitudes: np.array
        It contains the flattened n x m matrix where the altitudes of all
        the points in the map are stored
    step_size: tuple
        step sizes along each direction to create a linearly spaced grid
    h: float
        Safety threshold

    Returns
    ------
    S_hat: np.array
        Boolean array n_states x (n_actions + 1).
    """

    # Initialize
    n, m = world_shape
    n_states = n * m
    S_hat = np.zeros((n_states, n_actions + 1), dtype=bool)

    # In case an initial state is given
    if not np.isnan(s):
        S_hat[s, 0] = True
        valid_initial_seed = False
        vertical = False
        horizontal = False
        altitude_prev = altitudes[s]
        if not isinstance(s, np.ndarray):
            s = np.array([s])

        # Loop through actions
        for action in range(1, n_actions + 1):

            # Compute next state to check steepness
            next_vec_ind = _dynamics_vec_ind(s, action, world_shape)
            altitude_next = altitudes[next_vec_ind]

            if s != next_vec_ind and -np.abs(altitude_prev - altitude_next) / \
                    step_size[0] >= h:
                S_hat[s, action] = True
                S_hat[next_vec_ind, 0] = True
                S_hat[next_vec_ind, _reverse_action(action)] = True
                if action == 1 or action == 3:
                    vertical = True
                if action == 2 or action == 4:
                    horizontal = True

        if vertical and horizontal:
            valid_initial_seed = True

        if valid_initial_seed:
            return S_hat
        else:
            S_hat[:] = False
            return S_hat

    # If an explicit initial state is not given
    else:
        while np.all(np.logical_not(S_hat)):
            initial_state = np.random.choice(n_states)
            S_hat = compute_S_hat0(initial_state, world_shape, n_actions,
                                   altitudes, step_size, h)
        return S_hat

def compute_true_S_hat(graph, safe_set, initial_nodes, reverse_graph=None):
    """
    Compute the true safe set with reachability and returnability.

    Parameters
    ----------
    graph: nx.DiGraph
    safe_set: np.array
    initial_nodes: list of int
    reverse_graph: nx.DiGraph
        graph.reverse()

    Returns
    -------
    true_safe: np.array
        Boolean array n_states x (n_actions + 1).
    """
    graph = graph.copy()
    link_graph_and_safe_set(graph, safe_set)
    if reverse_graph is None:
        reverse_graph = graph.reverse()
    reach = reachable_set(graph, initial_nodes)
    ret = returnable_set(graph, reverse_graph, initial_nodes)
    ret &= reach
    return ret

def draw_gp_sample(kernel, world_shape, step_size):
    """
    Draws a sample from a Gaussian process distribution over a user
    specified grid

    Parameters
    ----------
    kernel: GPy kernel
        Defines the GP we draw a sample from
    world_shape: tuple
        Shape of the grid we use for sampling
    step_size: tuple
        Step size along any axis to find linearly spaced points
    """
    # Compute linearly spaced grid
    coord = _grid(world_shape, step_size)

    # Draw a sample from GP
    cov = kernel.K(coord) + np.eye(coord.shape[0]) * 1e-10
    sample = np.random.multivariate_normal(np.zeros(coord.shape[0]), cov)
    return sample, coord

class MultiagentGridWorldAgent(SafeMDP):
    """
    Multiagent Grid World with Safe Exploration

    Parameters:
    gp:          GPy.core.GP
                 Gaussian process that expresses our current belief over the safety
                 feature

    others_gp: list of GPy.core.GP
                 Gaussian process for action distribution of other agents

    world_shape: tuple<float, float>
                 n x m gridworld dimensions

    step_size:   tuple<float, float>
                 discretization along each axis

    beta:        float
                 Scaling factor to determine the amplitude of the confidence
                 intervals

    altitudes:   np.array
                 It contains the flattened n x m matrix where the altitudes
                 of all the points in the map are stored

    h:           float
                 Safety threshold

    S0:          np.array
                 n_states x (n_actions + 1) array of booleans that indicates which
                 states (first column) and which state-action pairs belong to the
                 initial safe seed. Notice that, by convention we initialize all
                 the states to be safe

    S_hat0:      np.array or nan
                 n_states x (n_actions + 1) array of booleans that indicates which
                 states (first column) and which state-action pairs belong to the
                 initial safe seed and satisfy recovery and reachability properties.
                 If it is nan, such a boolean matrix is computed during
                 initialization

    noise:       float
                 Standard deviation of the measurement noise

    L:           float
                 Lipschitz constant to compute expanders

    update_dist: int
                 Distance in unweighted graph used for confidence interval update.
                 A sample will only influence other nodes within this distance.

    other_pos:     np.array
                 k - 1 agents' positions in the gridworld
    """
    def __init__(self, gp, others_explore_gp, others_rewards_gp, world_shape, step_size,
                beta, altitudes, h, collide_threshold, S0, S_hat0, my_pos_ind, L,
                other_pos, epsilons, update_dist=0, gamma=0.9):

        self.S = S0.copy()
        graph = self.grid_world_graph(world_shape)
        link_graph_and_safe_set(graph, self.S)
        super(MultiagentGridWorldAgent, self).__init__(graph, gp, S_hat0, h, L, beta=2)

        self.altitudes = altitudes
        self.world_shape = world_shape
        self.step_size = step_size
        self.update_dist = update_dist
        self.others_explore_gp = others_explore_gp
        self.others_rewards_gp = others_rewards_gp
        self.gamma = gamma

        self.coord = _grid(self.world_shape, self.step_size)

        self.distance_matrix = cdist(self.coord, self.coord)

        # Confidence intervals
        self.l = np.empty(self.S.shape, dtype=float)
        self.u = np.empty(self.S.shape, dtype=float)
        self.l[:] = -np.inf
        self.u[:] = np.inf
        self.l[self.S] = h
        self.my_pos_ind = my_pos_ind

        self.num_other_agents = len(other_pos)
        self.other_pos = other_pos

        self.collide_threshold = collide_threshold
        self.epsilons = epsilons

        states_ind = np.arange(np.prod(self.world_shape))
        states_grid = states_ind.reshape(world_shape)

        self.action_move_dict = {
            0: np.array([0, 0]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
            4: np.array([0, 1]),
        }

        self._prev_up = states_grid[:, :-1].flatten()
        self._next_up = states_grid[:, 1:].flatten()
        self._prev_right = states_grid[:-1, :].flatten()
        self._next_right = states_grid[1:, :].flatten()

        self._mat_up = np.hstack((self.coord[self._prev_up, :],
                                  self.coord[self._next_up, :]))
        self._mat_right = np.hstack((self.coord[self._prev_right, :],
                                     self.coord[self._next_right, :]))

        self.other_best_actions = [0 for agent in range(self.num_other_agents)]
        self.other_agent_occupancy = [1.0 for i in range(self.world_shape[0] * self.world_shape[1])]
        self.prev_action = 0
        self.prev_node = self.my_pos_ind
        self.visited_node = set()

        self.trajs = [
            [_nodes_to_states(
                np.array([self.other_pos[agent]]),
                self.world_shape,
                self.step_size,
            )[0]] for agent in range(self.num_other_agents)
        ]

        self.other_value_functions = [
            [0.0 for i in range(self.world_shape[0] * self.world_shape[1])]
            for agent in range(self.num_other_agents)
        ]
        for agent in range(self.num_other_agents):
            self._value_iteration(agent)

    def plot_S(self, safe_set, action=0):
        """
        Plot the set of safe states

        Parameters
        ----------
        safe_set: np.array(dtype=bool)
            n_states x (n_actions + 1) array of boolean values that indicates
            the safe set
        action: int
            The action for which we want to plot the safe set.
        """
        plt.figure(action)
        plt.imshow(np.reshape(safe_set[:, action], self.world_shape).T,
                   origin='lower', interpolation='nearest', vmin=0, vmax=1)
        plt.title('action {0}'.format(action))
        plt.show()

    def update_confidence_interval(self, jacobian=False):
        """
        Updates the lower and the upper bound of the confidence intervals
        using then posterior distribution over the gradients of the altitudes

        Returns
        -------
        l: np.array
            lower bound of the safety feature (mean - beta*std)
        u: np.array
            upper bound of the safety feature (mean + beta*std)
        """

        # Initialize to unsafe
        self.l[:] = self.u[:] = self.h - 1

        # States are always safe
        self.l[:, 0] = self.u[:, 0] = self.h

        # Actions up and down
        mu_up, s_up = self.gp.predict(self._mat_up,
                                      kern=DifferenceKernel(self.gp.kern),
                                      full_cov=False)
        s_up = self.beta * np.sqrt(s_up)

        self.l[self._prev_up, 1, None] = mu_up - s_up
        self.u[self._prev_up, 1, None] = mu_up + s_up

        self.l[self._next_up, 3, None] = -mu_up - s_up
        self.u[self._next_up, 3, None] = -mu_up + s_up

        # Actions left and right
        mu_right, s_right = self.gp.predict(self._mat_right,
                                            kern=DifferenceKernel(
                                                self.gp.kern),
                                            full_cov=False)
        s_right = self.beta * np.sqrt(s_right)
        self.l[self._prev_right, 2, None] = mu_right - s_right
        self.u[self._prev_right, 2, None] = mu_right + s_right

        self.l[self._next_right, 4, None] = -mu_right - s_right
        self.u[self._next_right, 4, None] = -mu_right + s_right

        # Other agents position
        for agent in range(self.num_other_agents):
            agent_pos_ind = self.other_pos[agent]
            agent_coord = _nodes_to_states(
                np.array([agent_pos_ind]),
                self.world_shape,
                self.step_size,
            )[0]

            epsilon = self.epsilons[agent]

            best_next_state = agent_coord
            best_value = -float('inf')
            for action in range(0, 5):
                next_state = self.move_coordinate(agent_coord, action)
                next_node = _states_to_nodes(
                    np.array([next_state]),
                    self.world_shape,
                    self.step_size
                )[0]
                if self.other_value_functions[agent][next_node] > best_value:
                    best_next_state = next_state
                    best_value = self.other_value_functions[agent][next_node]

            self.update_occupancy(1.0 - epsilon, 0.0, 1.0, best_next_state)

            mu_explore_stay, s_explore_stay = self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], agent_coord[0], agent_coord[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            # mu_exploit_stay, s_exploit_stay = self.others_exploit_gp[agent].predict(
            #     np.array([[agent_coord[0], agent_coord[1], agent_coord[0], agent_coord[1]]]),
            #     kern=self.others_explore_gp[agent].kern,
            #     full_cov=False
            # )

            mu_stay = epsilon * mu_explore_stay
            s_stay = epsilon * s_explore_stay

            agent_up_coord = self.move_coordinate(agent_coord, 1)
            mu_explore_up, s_explore_up = self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], agent_up_coord[0], agent_up_coord[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            # mu_exploit_up, s_exploit_up = self.others_exploit_gp[agent].predict(
            #     np.array([[agent_coord[0], agent_coord[1], agent_up_coord[0], agent_up_coord[1]]]),
            #     kern=self.others_explore_gp[agent].kern,
            #     full_cov=False
            # )
            mu_up = epsilon * mu_explore_up
            s_up = epsilon * s_explore_up

            agent_down_coord = self.move_coordinate(agent_coord, 3)
            mu_explore_down, s_explore_down = self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], agent_down_coord[0], agent_down_coord[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            # mu_exploit_down, s_exploit_down = self.others_exploit_gp[agent].predict(
            #     np.array([[agent_coord[0], agent_coord[1], agent_down_coord[0], agent_down_coord[1]]]),
            #     kern=self.others_explore_gp[agent].kern,
            #     full_cov=False
            # )
            mu_down = epsilon * mu_explore_down
            s_down = epsilon * s_explore_down

            agent_left_coord = self.move_coordinate(agent_coord, 2)
            mu_explore_left, s_explore_left = self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], agent_left_coord[0], agent_left_coord[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            # mu_exploit_left, s_exploit_left = self.others_exploit_gp[agent].predict(
            #     np.array([[agent_coord[0], agent_coord[1], agent_left_coord[0], agent_left_coord[1]]]),
            #     kern=self.others_explore_gp[agent].kern,
            #     full_cov=False
            # )
            mu_left = epsilon * mu_explore_left
            s_left = epsilon * s_explore_left

            agent_right_coord = self.move_coordinate(agent_coord, 4)
            mu_explore_right, s_explore_right = self.others_explore_gp[agent].predict(
                np.array([[agent_coord[0], agent_coord[1], agent_right_coord[0], agent_right_coord[1]]]),
                kern=self.others_explore_gp[agent].kern,
                full_cov=False
            )
            # mu_exploit_right, s_exploit_right = self.others_exploit_gp[agent].predict(
            #     np.array([[agent_coord[0], agent_coord[1], agent_right_coord[0], agent_right_coord[1]]]),
            #     kern=self.others_explore_gp[agent].kern,
            #     full_cov=False
            # )
            mu_right = epsilon * mu_explore_right
            s_right = epsilon * s_explore_right

            scale = 1 / (max(exp(mu_stay) + exp(mu_up) + exp(mu_down) + exp(mu_left) + exp(mu_down), 1e-2))

            self.update_occupancy(mu_stay, s_stay, scale, agent_coord)

            self.update_occupancy(mu_up, s_up, scale, agent_up_coord)

            self.update_occupancy(mu_down, s_down, scale, agent_down_coord)

            self.update_occupancy(mu_left, s_left, scale, agent_left_coord)

            self.update_occupancy(mu_right, s_right, scale, agent_right_coord)

    def update_occupancy(self, mu, s, scale, coord):
        prob = exp(mu + s)
        ind = _states_to_nodes(
            np.array([coord]),
            self.world_shape,
            self.step_size
        )[0]
        self.other_agent_occupancy[ind] *= (1 - prob * scale)

    def grid_world_graph(self, world_size):
        """Create a graph that represents a grid world.

        In the grid world there are four actions, (1, 2, 3, 4), which correspond
        to going (up, right, down, left) in the x-y plane. The states are
        ordered so that `np.arange(np.prod(world_size)).reshape(world_size)`
        corresponds to a matrix where increasing the row index corresponds to the
        x direction in the graph, and increasing y index corresponds to the y
        direction.

        Parameters
        ----------
        world_size: tuple
                    The size of the grid world (rows, columns)

        Returns
        -------
        graph:      nx.DiGraph()
                    The directed graph representing the grid world.
        """
        nodes = np.arange(np.prod(world_size))
        grid_nodes = nodes.reshape(world_size)

        graph = nx.DiGraph()

        # action 1: go right
        graph.add_edges_from(zip(grid_nodes[:, :-1].reshape(-1),
                                 grid_nodes[:, 1:].reshape(-1)),
                             action=2)

        # action 2: go down
        graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
                                 grid_nodes[1:, :].reshape(-1)),
                             action=3)

        # action 3: go left
        graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
                                 grid_nodes[:, :-1].reshape(-1)),
                             action=4)

        # action 4: go up
        graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
                                 grid_nodes[:-1, :].reshape(-1)),
                             action=1)
        print('Edges Added')
        return graph

    def target_sample(self):
        """
        Compute the next target (s, a) to sample (highest uncertainty within
        G or S_hat)

        Returns
        -------
        node: int
            The next node to sample
        action: int
            The next action to sample
        """
        uncertainties = []
        for action in range(0, 5):
            uncertainty = self.u[self.my_pos_ind, action, None] - self.l[self.my_pos_ind, action, None]
            uncertainties += [(action, uncertainty)]

        uncertainties.sort(key=lambda x: x[1], reverse=True)

        my_coord = _nodes_to_states(
            np.array([self.my_pos_ind]),
            self.world_shape,
            self.step_size,
        )[0]

        for uncertainty in uncertainties:
            new_coord = self.move_coordinate(my_coord, uncertainty[0])
            new_ind = _states_to_nodes(
                np.array([new_coord]),
                self.world_shape,
                self.step_size,
            )[0]
            if self.other_agent_occupancy[new_ind] > self.collide_threshold \
                    and np.any(self.l[new_ind, :, None] > self.h) \
                    and self.returnable(new_ind) \
                    and (new_ind not in self.visited_node):
                self.prev_node = self.my_pos_ind
                self.my_pos_ind = new_ind
                self.prev_action = uncertainty[0]
                self.other_agent_occupancy = [1.0 for i in range(self.world_shape[0] * self.world_shape[1])]
                self.visited_node.add(new_ind)
                return new_ind, uncertainty[0], new_coord

        for uncertainty in uncertainties:
            new_coord = self.move_coordinate(my_coord, uncertainty[0])
            new_ind = _states_to_nodes(
                np.array([new_coord]),
                self.world_shape,
                self.step_size,
            )[0]
            if self.other_agent_occupancy[new_ind] > self.collide_threshold \
                    and np.any(self.l[new_ind, :, None] > self.h) \
                    and self.returnable(new_ind):
                self.prev_node = self.my_pos_ind
                self.my_pos_ind = new_ind
                self.prev_action = uncertainty[0]
                self.other_agent_occupancy = [1.0 for i in range(self.world_shape[0] * self.world_shape[1])]
                self.visited_node.add(new_ind)
                return new_ind, uncertainty[0], new_coord

        self.other_agent_occupancy = [1.0 for i in range(self.world_shape[0] * self.world_shape[1])]
        new_ind = self.prev_node
        self.prev_node = self.my_pos_ind
        self.my_pos_ind = new_ind
        self.prev_action = _reverse_action(self.prev_action)
        self.visited_node.add(new_ind)
        return self.my_pos_ind, self.prev_action, my_coord

    def returnable(self, node):
        coord = _nodes_to_states(
            np.array([node]),
            self.world_shape,
            self.step_size,
        )[0]
        for action in range(0, 5):
            new_coord = self.move_coordinate(coord, action)
            new_ind = _states_to_nodes(
                np.array([new_coord]),
                self.world_shape,
                self.step_size,
            )[0]
            if np.any(self.S[new_ind, :]):
                return True

        return False

    def update_sets(self):
        """
        Update the sets S, S_hat and G taking with the available observation
        """
        self.update_confidence_interval()
        # self.S[:] = self.l >= self.h
        self.S |= self.l >= self.h

        self.compute_S_hat()

    def add_observation(self, node, action, others_actions, others_rewards):
        """
        Add an observation of the given state-action pair.

        Observing the pair (s, a) means adding an observation of the altitude
        at s and an observation of the altitude at f(s, a)

        Parameters
        ----------
        node: int
            Node index
        action: int
            Action index
        """
        # Observation of next state
        for _, next_node, data in self.graph.out_edges(node, data=True):
            if data['action'] == action:
                break

        self.add_gp_observations(self.coord[[node, next_node], :],
                                 self.altitudes[[node, next_node], None])

        for agent in range(self.num_other_agents):
            other_pos_coord = _nodes_to_states(
                np.array([self.other_pos[agent]]),
                self.world_shape,
                self.step_size,
            )[0]

            for action in range(0, 5):
                other_new_coord = other_pos_coord
                if action != 0:
                    other_new_coord = self.move_coordinate(other_pos_coord, action)
                self.others_rewards_gp[agent].set_XY(
                    [
                        [
                            other_new_coord[0],
                            other_new_coord[1]
                        ]
                    ],
                    [[others_rewards[agent]]]
                )
                self._value_iteration(agent)

                self.others_explore_gp[agent].set_XY(
                    [
                        [
                            other_pos_coord[0],
                            other_pos_coord[1],
                            other_new_coord[0],
                            other_new_coord[1]
                        ]
                    ],
                    [[float(action != others_actions[agent])]]
                )

            other_pos_coord_new = self.move_coordinate(other_pos_coord, others_actions[agent])
            self.trajs[agent].append(other_pos_coord_new)

            other_pos_ind = _states_to_nodes(
                np.array([other_pos_coord_new]),
                self.world_shape,
                self.step_size
            )[0]

            self.other_pos[agent] = other_pos_ind

            self.epsilons[agent] = self.optimize_for_epsilon(agent)

    def optimize_for_epsilon(self, agent):
        traj = self.trajs[agent]
        def _compute_log_likelihood(epsilon):
            sum_log_likelihood = 0.
            for step in range(1, len(traj)):
                explore_prob, _ = self.others_explore_gp[agent].predict(
                    np.array([[traj[step - 1][0], traj[step - 1][1], traj[step][0], traj[step][1]]]),
                    kern=self.others_explore_gp[agent].kern,
                    full_cov=False
                )

                scale_denominator = 0.0
                for action in range(5):
                    new_coord = self.move_coordinate(traj[step - 1], action)
                    prob, _ = self.others_explore_gp[agent].predict(
                        np.array([[traj[step - 1][0], traj[step - 1][1], new_coord[0], new_coord[1]]]),
                        kern=self.others_explore_gp[agent].kern,
                        full_cov=False
                    )
                    scale_denominator += exp(prob)

                cur_coord = [traj[step - 1][0], traj[step - 1][1]]
                best_next_state = cur_coord
                best_value = -float('inf')
                for action in range(0, 5):
                    next_state = self.move_coordinate(cur_coord, action)
                    next_node = _states_to_nodes(
                        np.array([next_state]),
                        self.world_shape,
                        self.step_size
                    )[0]
                    if self.other_value_functions[agent][next_node] > best_value:
                        best_next_state = next_state
                        best_value = self.other_value_functions[agent][next_node]
                try:
                    if epsilon[0] == 0.0:
                        epsilon[0] = 1e-6
                    sum_log_likelihood += log(epsilon[0] * exp(explore_prob) / scale_denominator + (1 - epsilon[0]) * float(best_next_state[0] == traj[step][0] and best_next_state[1] == traj[step][1]))
                except:
                    print(epsilon[0], explore_prob, scale_denominator, best_next_state, traj[step])

            return -sum_log_likelihood

        res = minimize(_compute_log_likelihood, np.array([0.5]), method='L-BFGS-B', bounds=np.array([(0.0, 1.0)]))
        return res.x[0]

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

    def _value_iteration(self, agent):
        difference = 10000

        while difference > 0.01:
            cur_difference = 0.0
            num_nodes = self.world_shape[0] * self.world_shape[1]
            for node in range(num_nodes):
                state = _nodes_to_states(
                    np.array([node]),
                    self.world_shape,
                    self.step_size,
                )[0]
                cur_reward, _ = self.others_rewards_gp[agent].predict(
                    np.array([[state[0], state[1]]]),
                    kern=self.others_rewards_gp[agent].kern,
                    full_cov=False
                )
                old_value = self.other_value_functions[agent][node]
                for action in range(0, 5):
                    new_state = self.move_coordinate(state, action)
                    new_node = _states_to_nodes(
                        np.array([new_state]),
                        self.world_shape,
                        self.step_size
                    )[0]
                    self.other_value_functions[agent][node] = max(
                        cur_reward + self.gamma * self.other_value_functions[agent][new_node],
                        old_value
                    )

                cur_difference = max(
                    cur_difference,
                    abs(self.other_value_functions[agent][node] - old_value)
                )

            difference = cur_difference

def _grid(world_shape, step_size):
    """
    Creates grids of coordinates and indices of state space

    Parameters
    ----------
    world_shape: tuple
                 Size of the grid world (rows, columns)
    step_size:   tuple
                 Phyiscal step size in the grid world

    Returns
    -------
    states_ind:  np.array
                 (n*m) x 2 array containing the indices of the states
    states_coord: np.array
                  (n*m) x 2 array containing the coordinates of the states
    """
    nodes = np.arange(0, world_shape[0] * world_shape[1])
    return _nodes_to_states(nodes, world_shape, step_size)

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

def _reverse_action(action):
    # Computes the action that is the opposite of the one given as input

    rev_a = np.mod(action + 2, 4)
    if rev_a == 0:
        rev_a = 4
    return rev_a

def _dynamics_vec_ind(states_vec_ind, action, world_shape):
    """
    Dynamic evolution of the system defined in vector representation of
    the states

    Parameters
    ----------
    states_vec_ind: np.array
        Contains all the vector indexes of the states we want to compute
        the dynamic evolution for
    action: int
        action performed by the agent

    Returns
    -------
    next_states_vec_ind: np.array
        vector index of states resulting from applying the action given
        as input to the array of starting points given as input
    """
    n, m = world_shape
    next_states_vec_ind = np.copy(states_vec_ind)
    if action == 1:
        next_states_vec_ind[:] = states_vec_ind + 1
        condition = np.mod(next_states_vec_ind, m) == 0
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 2:
        next_states_vec_ind[:] = states_vec_ind + m
        condition = next_states_vec_ind >= m * n
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 3:
        next_states_vec_ind[:] = states_vec_ind - 1
        condition = np.mod(states_vec_ind, m) == 0
        next_states_vec_ind[condition] = states_vec_ind[condition]
    elif action == 4:
        next_states_vec_ind[:] = states_vec_ind - m
        condition = next_states_vec_ind <= -1
        next_states_vec_ind[condition] = states_vec_ind[condition]
    else:
        raise ValueError("Unknown action")
    return next_states_vec_ind
