from __future__ import division, print_function, absolute_import

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

from utilities import DifferenceKernel
from SafeMDP_class import (reachable_set, returnable_set, SafeMDP,
                            link_graph_and_safe_set)

class MultiagentGridWorldAgent(SafeMDP):
    """
    Multiagent Grid World with Safe Exploration

    Parameters:
    gp:          GPy.core.GP
                 Gaussian process that expresses our current belief over the safety
                 feature

    gp_epsilons: GPy.core.GP
                 Gaussian process for epsilons of other agents

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
    def __init__(self, gp, gp_epsilons, world_shape, step_size, beta, altitudes, h, S0,
                S_hat0, L, update_dist=0, other_pos=None):

        self.S = S0.copy()
        graph = grid_world_graph(world_shape)
        link_graph_and_safe_set(graph, self.S)
        super(GridWorld, self).__init__(graph, gp, S_hat0, h, L, beta=2)

        self.altitudes = altitudes
        self.world_shape = world_shape
        self.step_size = step_size
        self.update_dist = update_dist

        self.coord = _grid(self.world_shape, self.step_size)

        self.distance_matrix = cdist(self.coord, self.coord)

        # Confidence intervals
        self.l = np.empty(self.S.shape, dtype=float)
        self.u = np.empty(self.S.shape, dtype=float)
        self.l[:] = -np.inf
        self.u[:] = np.inf
        self.l[self.S] = h

        self.num_other_agents = len(other_pos)
        self.other_pos = other_pos

        states_ind = np.arange(np.prod(self.world_shape))
        states_grid = states_ind.reshape(world_shape)

        self._prev_up = states_grid[:, :-1].flatten()
        self._next_up = states_grid[:, 1:].flatten()
        self._prev_right = states_grid[:-1, :].flatten()
        self._next_right = states_grid[1:, :].flatten()

        self._mat_up = np.hstack((self.coord[self._prev_up, :],
                                  self.coord[self._next_up, :]))
        self._mat_right = np.hstack((self.coord[self._prev_right, :],
                                     self.coord[self._next_right, :]))

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

    def grid_world_graph(world_size):
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
                             action=1)

        # action 2: go down
        graph.add_edges_from(zip(grid_nodes[:-1, :].reshape(-1),
                                 grid_nodes[1:, :].reshape(-1)),
                             action=2)

        # action 3: go left
        graph.add_edges_from(zip(grid_nodes[:, 1:].reshape(-1),
                                 grid_nodes[:, :-1].reshape(-1)),
                             action=3)

        # action 4: go up
        graph.add_edges_from(zip(grid_nodes[1:, :].reshape(-1),
                                 grid_nodes[:-1, :].reshape(-1)),
                             action=4)
        print('Edges Added')
        return graph


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
    return nodes_to_states(nodes, world_shape, step_size)

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
