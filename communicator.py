# This code is meant to improve existing code which comes from
# https://github.com/epfml/ChocoSGD/tree/master/convex_code.
# As such, some of the code below is taken directly from that repository,
# and the credits should go to its authors.

import networkx as nx
import numpy as np


class Communicator:
    """Class that encapsulates all attributes and methods needed to perform the communication part
    of the decentralized SGD."""
    def __init__(self, method='plain', n_machines=1, topology='complete', consensus_lr=1.0):
        """Constructor for the Communicator class."""
        self.method = method
        self.n_machines = n_machines
        self.topology = topology
        self.consensus_lr = consensus_lr

        # Validate the input parameters
        self.__validate_params()

        # Set parameters that need to be computed (transition matrix)
        self.__set_params()

    def __set_params(self):
        """Compute extra parameters that depend on the input parameters."""

        # Create a valid transition matrix depending on the chosen topology
        self.transition_matrix = self.__create_valid_transition_matrix()

    def __validate_params(self):
        """Validate input parameters."""

        # Define the supported methods for communication
        valid_methods = ['plain', 'choco']

        # Check if method is valid
        if self.method not in valid_methods:
            raise ValueError('Method for communication should be one of: ' + str(valid_methods))

        # If "plain" method is used, make sure consensus learning rate is set to 1
        if self.method == 'plain':
            self.consensus_lr = 1.0

        # Check that consensus learning rate is set when using choco method
        if self.method == 'choco' and not self.consensus_lr:
            raise ValueError('Method for communication "Choco" should be given parameter "consensus_lr"')

        # Check if number of machines is an integer
        if not isinstance(self.n_machines, int):
            raise ValueError('Invalid number of machines specified, must be an integer')

        # Check if number of machines is a positive value
        if self.n_machines <= 0:
            raise ValueError('Invalid number of machines specified, must be a positive integer')

        # Check if consensus lr is decimal (or an integer that is 0 or 1)
        if not isinstance(self.consensus_lr, float):

            if not isinstance(self.consensus_lr, int) or self.consensus_lr != 0 or self.consensus_lr != 1:
                raise ValueError('Invalid consensus learning rate, must be a (real) number')

        # Check if consensus lr is in interval [0,1]
        if not 0 <= self.consensus_lr <= 1:
            raise ValueError('Invalid consensus learning rate, must be in interval [0,1]')

        # Define the supported topologies
        valid_topologies = ['disconnected', 'path', 'star', 'ring', 'complete', 'barbell', 'torus']

        # Check if given String for topology is valid
        if isinstance(self.topology, str):
            if self.topology not in valid_topologies:
                raise ValueError('Topology should be one of: ' + str(valid_topologies))

            # If barbell topolgy, number of machines must be even
            if self.topology == 'barbell' and self.n_machines % 2 != 0:
                raise ValueError('For barbell topology, number of machines must be even')

            # If torus topology, number of machines must be a perfect square
            if self.topology == 'torus' and int(np.sqrt(self.n_machines)) ** 2 != self.n_machines:
                raise ValueError('For torus topology, number of machines must be a perfect square')

        # Check if given transition matrix for topology is valid
        elif isinstance(self.topology, np.ndarray):

            # Get dimensions of the transition matrix
            n_rows, n_cols = self.topology.shape

            # Check if the transition matrix is square
            if self.topology.ndim != 2 or n_rows != n_cols:
                raise ValueError('Invalid dimensions for transition matrix for topology')

            # Check if the transition matrix is irreducible (forms a strongly connected graph)
            if not nx.is_connected(nx.from_numpy_array(self.topology)):
                raise ValueError('Invalid transition matrix, must be irreducible (strongly connected graph)')

            # Check if number of machines indicated corresponds to the number of machines
            # induced by the given transition matrix
            if n_rows != self.n_machines:
                self.n_machines = n_rows
                print('n_machines does not correspond to the shape of the transition matrix, n_machines was updated to be correct')

        else:
            raise ValueError('Invalid type for topology')

    def __create_valid_transition_matrix(self):

        # If the transition matrix is to be generated given a String
        if isinstance(self.topology, str):

            # Get corresponding adjacency matrix
            if self.topology == 'disconnected':
                adjacency_matrix = np.eye(self.n_machines)
            elif self.topology == 'path':
                adjacency_matrix = np.eye(self.n_machines)
                np.fill_diagonal(adjacency_matrix[1:], val=1.0, wrap=False)
                np.fill_diagonal(adjacency_matrix[:, 1:], val=1.0, wrap=False)
            elif self.topology == 'star':
                adjacency_matrix = np.eye(self.n_machines)
                adjacency_matrix[0, :] = np.ones(self.n_machines)
                adjacency_matrix[:, 0] = np.ones(self.n_machines)
            elif self.topology == 'ring':
                adjacency_matrix = np.eye(self.n_machines)
                np.fill_diagonal(adjacency_matrix[1:], val=1.0, wrap=False)
                np.fill_diagonal(adjacency_matrix[:, 1:], val=1.0, wrap=False)
                adjacency_matrix[0, n_machines - 1] = 1.0
                adjacency_matrix[n_machines - 1, 0] = 1.0
            elif self.topology == 'complete':
                adjacency_matrix = np.ones((self.n_machines, self.n_machines))
            elif self.topology == 'barbell':
                adjacency_matrix = np.eye(self.n_machines)
                half_machines = self.n_machines // 2
                adjacency_matrix[:half_machines, :half_machines] = np.ones((half_machines, half_machines))
                adjacency_matrix[half_machines:, half_machines:] = np.ones((half_machines, half_machines))
                adjacency_matrix[half_machines, half_machines - 1] = 1.0
                adjacency_matrix[half_machines - 1, half_machines] = 1.0
            elif self.topology == 'torus':
                # Number of machines on "side" of torus
                n_machines_on_side = int(np.sqrt(self.n_machines))
                graph = nx.generators.lattice.grid_2d_graph(n_machines_on_side, n_machines_on_side, periodic=True)

                # Get the adjacency matrix from the graph
                adjacency_matrix = nx.adjacency_matrix(graph).toarray()

                # Add self-loops
                adjacency_matrix = adjacency_matrix + np.eye(self.n_machines)

        # Else the transition matrix is directly given in parameter "topology"
        else:
            adjacency_matrix = self.topology

        # Make sure that the rows of the matrix are normalized
        if not np.allclose(adjacency_matrix.sum(axis=1), np.ones(self.n_machines)):

            # Normalize each row to be a valid Markov Chain
            row_sums = adjacency_matrix.sum(axis=1, keepdims=True)
            transition_matrix = adjacency_matrix / row_sums

        # Else the adjacency matrix is already a valid transition matrix
        else:
            transition_matrix = adjacency_matrix

        return transition_matrix

    def communicate(self, weight_matrix, weight_matrix_hat):
        """Perform the communication step of the decentralized SGD, given the weight of
        all the machines as a matrix, using the Choco update"""
        return weight_matrix + self.consensus_lr * (weight_matrix_hat @ (self.transition_matrix - np.eye(self.n_machines)))
