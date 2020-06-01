from abc import ABC, abstractmethod
import numpy as np
import time

from communicator import Communicator
from quantizer import Quantizer
from print_util import *
INIT_WEIGHT_STD = 1


class DecentralizedSGDClassifier(ABC):
    """Abstract class that encapsulates all attributes and methods needed to perform the decentralized SGD."""

    def __init__(self, num_epoch=1,
                 initial_lr=0.1,
                 lr_type='bottou',
                 tau=None,
                 tol=1e-3,
                 regularizer=1e-4,
                 n_machines=1,
                 topology='complete',
                 communication_method='plain',
                 consensus_lr=1.0,
                 communication_frequency=1,
                 data_distribution_strategy='naive',
                 data_distribution_random_seed=None,
                 quantization_method="full",
                 features_to_keep=None,
                 random_seed=None,
                 compute_loss_every=50
                 ):
        """Constructor for the DecentralizedSGDClassifier class."""

        self.quantizer = Quantizer(quantization_method, features_to_keep)
        self.communicator = Communicator(communication_method, n_machines, topology, consensus_lr)

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.tol = tol
        self.regularizer = regularizer
        self.tau = tau
        self.communication_frequency = communication_frequency
        self.random_seed = random_seed
        self.data_distribution_strategy = data_distribution_strategy
        self.data_distribution_random_seed = data_distribution_random_seed
        self.compute_loss_every = compute_loss_every
        self.is_fitted = False

        # Validate the input parameters
        self.__validate_params()

    def __validate_params(self):
        """Validate input parameters."""

        need_positive_initial_lr = ['constant', 'decay']

        # Check positive initial_lr if need lr_type is in ['constant', 'decay']
        if self.lr_type in need_positive_initial_lr and not self.initial_lr > 0:
            raise ValueError('If lr_type is in ' + str(need_positive_initial_lr) + 'initial_lr must be > 0')

        # Check that if lr_type is decay, then parameters initial_lr and tau are given and regularizer is > 0
        if self.lr_type == 'decay':
            if not self.initial_lr:
                raise ValueError('If lr_type is decay, parameter initial_lr should be given')
            if not self.tau:
                raise ValueError('If lr_type is decay, parameter tau should be given')
            if not self.regularizer > 0:
                raise ValueError('If lr_type is decay, parameter regularizer must be strictly superior to 0')

        valid_data_distribution_strategy = ['undistributed', 'naive', 'random', 'label-sorted']

        # Check that if the data are distributed, then there is a valid data distribution strategy
        if self.data_distribution_strategy not in valid_data_distribution_strategy:
            raise ValueError('Inavlid data distribution strategy, value must be one of' + str(valid_data_distribution_strategy))

    def __update_lr(self, curr_iteration):
        """Compute the learning rate at the given epoch and iteration."""

        if self.lr_type == 'constant':
            return self.initial_lr
        if self.lr_type == 'decay':
            return self.initial_lr / (self.regularizer * (curr_iteration + self.tau))
        if self.lr_type == 'bottou':
            return self.initial_lr / (1 + self.initial_lr * self.regularizer * curr_iteration)

    def __distribute_data(self, y):
        """Distribute the data onto machines following the data distribution strategy."""
        num_samples = len(y)

        if self.data_distribution_strategy != 'undistributed':
            np.random.seed(self.data_distribution_random_seed)
            if self.data_distribution_strategy == 'random':
                all_indexes = np.arange(num_samples)
                np.random.shuffle(all_indexes)
            elif self.data_distribution_strategy == 'naive':
                all_indexes = np.arange(num_samples)
            elif self.data_distribution_strategy == 'label-sorted':
                all_indexes = np.argsort(y)

            # The number of samples per machine
            num_samples_per_machine = num_samples // self.communicator.n_machines

            # Create list of list containing indices of datapoints for each machine
            indices = [all_indexes[i:i + num_samples_per_machine] for i in range(0, len(all_indexes), num_samples_per_machine)]

            # Delete extra data if there is
            if not (num_samples / self.communicator.n_machines).is_integer():
                del indices[-1]
        else:
            num_samples_per_machine = num_samples
            indices = np.tile(np.arange(num_samples), (self.communicator.n_machines, 1))

        return indices, num_samples_per_machine

    @abstractmethod
    def loss(self, A, y):
        """Compute the loss.
        :param A: input data
        :param y: target data
        """
        pass

    @abstractmethod
    def gradient(self, A, y, sample_indices):
        """Compute the logistic loss gradient of the weights of each machine
        w.r.t. the chosen random sample at each machine
        :param A: input data
        :param y: target data
        :param sample_indices: indices of the selected sample for each machine
        """
        pass

    @abstractmethod
    def predict(self, A):
        """Predict target data of input data A using the fitted model.
        :param A: input data
        """
        pass

    def score(self, A, y):
        """Compute the accuracy of the model given input data A and target data y
        :param A: input data
        :param y: target data
        """
        # Get the prediction
        pred = self.predict(A)

        # Return the accuracy
        return np.mean(pred == y)

    def weight_matrix(self):
        """Return the matrix containing the weights learned by the model, where each
        column in the matrix corresponds to the weight vector of one machine"""
        return self.X if self.is_fitted else None

    def fit(self, A, y_init, logging=False):
        """Create the model using decentralized SGD on input data A and target data y
        :param A: input data
        :param y_init: target data
        """
        self.is_fitted = True

        # Make sure that labels are 0 and 1 and not -1 and 1
        y = 1 * (y_init > 0.0)

        num_samples, num_features = A.shape
        n_machines = self.communicator.n_machines

        np.random.seed(self.random_seed)

        # Initialization of the parameters
        self.X = np.random.normal(0, 0, size=(num_features,))
        self.X = np.tile(self.X, (n_machines, 1)).T
        X_hat = np.zeros_like(self.X)

        lr = self.initial_lr

        # Distribute the data onto the machines
        indices, num_samples_per_machine = self.__distribute_data(y)

        diff_losses = np.inf
        curr_loss = np.inf

        all_losses = np.zeros(int(np.ceil(num_samples_per_machine * self.num_epoch / self.compute_loss_every)))

        train_start = time.time()

        if logging:
            # Print logging header
            log_acc_loss_header(color=Color.GREEN)

        # Decentralized SGD
        for epoch in range(0, self.num_epoch):
            if diff_losses > self.tol:
                for iteration in range(num_samples_per_machine):

                    curr_iteration = epoch * num_samples_per_machine + iteration

                    # Select a random sample for each machine
                    sample_indices = [np.random.choice(indices[machine]) for machine in range(0, n_machines)]

                    # Gradient step
                    self.X -= lr * self.gradient(A, y, sample_indices)

                    # Communicate to neighbors and quantize
                    if curr_iteration % self.communication_frequency == 0:
                        # Communication step
                        self.X = self.communicator.communicate(self.X, X_hat)

                        # Quantization step
                        X_hat += self.quantizer.quantize(self.X - X_hat)

                    # Compute the new loss
                    new_loss = self.loss(A, y)

                    # Compute the difference between last loss and current (check tol)
                    diff_losses = np.abs(new_loss - curr_loss)

                    # Update current loss value to new one
                    curr_loss = new_loss

                    # Update learning rate
                    lr = self.__update_lr(curr_iteration)

                    if logging:
                        # Print iteration information

                        # Compute and print score only once per epoch
                        if iteration == num_samples_per_machine - 1:
                            score = self.score(A, y)
                        else:
                            score = None

                        # Print
                        log_acc_loss(epoch, self.num_epoch, iteration, num_samples_per_machine, time.time() - train_start, score, curr_loss, persistent=False)

                    if curr_iteration % self.compute_loss_every == 0:
                        all_losses[curr_iteration // self.compute_loss_every] = curr_loss

                    # If loss is infinite or NaN, stop the training
                    if np.isinf(curr_loss) or np.isnan(curr_loss):
                        print("Training interrupted, loss is either infinity or NaN")
                        break
            if logging:
                print()

        return all_losses
