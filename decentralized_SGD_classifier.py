from abc import ABC, abstractmethod
import numpy as np
import time

from communicator import Communicator
from quantizer import Quantizer

INIT_WEIGHT_STD = 1
LOSS_PER_EPOCH = 10


class DecentralizedSGDClassifier(ABC):
    """Abstract class that encapsulates all attributes and methods needed to perform the decentralized SGD."""

    def __init__(self, num_epoch,
                 lr_type,
                 initial_lr=None,
                 tol=0.00001,
                 regularizer=None,
                 epoch_decay_lr=None,
                 consensus_lr=None,
                 quantization="full", # Different from the quantizer => check with Paul TODO
                 # number of coordinates k in top-k or random-k quantization
                 coordinates_to_keep=None,
                 # number of levels in qsgd quantization
                 num_levels=None,
                 estimate='final',
                 n_machines=1,
                 topology='centralized',  # Different from the communicator => check with Adrien TODO
                 method='choco',  # Different from the communicator => check with Adrien TODO
                 distribute_data=False,
                 # whether each machine gets random data or continuous set of data
                 # might not have any difference, depends on the dataset
                 split_data_strategy=None,
                 tau=None,
                 real_update_every=1,
                 random_seed=None,
                 split_data_random_seed=None,
                 ):
        """Constructor for the DecentralizedSGDClassifier class."""

        assert estimate in ['final', 'mean', 't+tau', '(t+tau)^2'] # Not used yet TODO

        assert method in ['choco', 'dcd-psgd', 'ecd-psgd', 'plain'] # Where to deal with that ? TODO
        if method in ['dcd-psgd', 'ecd-psgd']:
            assert quantization in ['random-unbiased', 'qsgd-unbiased']

        self.quantizer = Quantizer(quantization, coordinates_to_keep, num_levels)
        self.communicator = Communicator(method, n_machines, topology, consensus_lr)

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.tol = tol
        self.regularizer = regularizer
        self.epoch_decay_lr = epoch_decay_lr
        self.estimate = estimate  # Not used yet TODO
        self.tau = tau
        self.real_update_every = real_update_every  # Not used yet TODO
        self.random_seed = random_seed
        self.distribute_data = distribute_data
        self.split_data_strategy = split_data_strategy
        self.split_data_random_seed = split_data_random_seed

        # Validate the input parameters
        self.__validate_params()

        self.X = None
        self.X_hat = None
        self.is_fitted = False

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

        # Check that if lr_type is epoch-delay, then epoch_decay_lr is given
        if self.lr_type == 'epoch-decay' and not self.epoch_decay_lr:
            raise ValueError('If lr_type is epoch-decay, parameter epoch_decay_lr should be given')

        valid_split_data_strategy = ['naive', 'random', 'label-sorted']

        # Check that if the data are distributed, then there is a valid split data strategy
        if self.distribute_data and not self.split_data_strategy in valid_split_data_strategy:
            raise ValueError('If the data are distributed, split_data_strategy must be not None')

    def __update_lr(self, curr_iteration):
        """Compute the learning rate at the given epoch and iteration.
        """

        if self.lr_type == 'constant':
            return self.initial_lr
        if self.lr_type == 'epoch-decay':
            return self.initial_lr * (self.epoch_decay_lr ** epoch)
        if self.lr_type == 'decay':
            return self.initial_lr / (self.regularizer * (curr_iteration + self.tau))
        if self.lr_type == 'bottou':
            return self.initial_lr / (1 + self.initial_lr * self.regularizer * curr_iteration)

    def __split_data(self, y):
        """Split the data onto machines following the split data strategy.
        """
        num_samples = len(y)

        if self.distribute_data:
            np.random.seed(self.split_data_random_seed)
            if self.split_data_strategy == 'random':
                all_indexes = np.arange(num_samples)
                np.random.shuffle(all_indexes)
            elif self.split_data_strategy == 'naive':
                all_indexes = np.arange(num_samples)
            elif self.split_data_strategy == 'label-sorted':
                all_indexes = np.argsort(y)

            # The number of samples per machine
            num_samples_per_machine = num_samples // self.communicator.n_machines

            # Create list of list containing indices of datapoints for each machine
            indices = [all_indexes[i:i + num_samples_per_machine] for i in range(0, len(all_indexes), num_samples_per_machine)]

            # Delete extra data if there is
            if not (num_samples / self.communicator.n_machines).is_integer():
                del indices[-1]

            print("length of indices:", len(indices))
            print("length of last machine indices:", len(indices[-1]))

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
        raise NotImplementedError("Abstract method")

    @abstractmethod
    def gradient(self, A, y, sample_indices):
        """Compute the gradient of the sample at index "sample_idx" of the input data "A"
        with the model of the machine number "machine" using target data "y".
        :param A: input data
        :param y: target data
        :param sample_indices: indices of the samples for each machine
        """
        pass

    @abstractmethod
    def predict(self, A):
        """Predict target data of input data A.
        :param A: input data
        """
        pass

    @abstractmethod
    def score(self, A, y):
        """Score in comparing predictions on data input A and target data y
        :param A: input data
        :param y: target data
        """
        pass

    def fit(self, A, y_init):
        """Create the model using decentralized SGD on input data A and target data y
        :param A: input data
        :param y_init: target data
        """
        self.is_fitted = True

        y = np.copy(y_init)
        num_samples, num_features = A.shape
        n_machines = self.communicator.n_machines
        losses = np.zeros(self.num_epoch + 1)

        # Initialization of the parameters
        if self.X is None:
            self.X = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
            self.X = np.tile(self.X, (n_machines, 1)).T
            self.X_hat = np.zeros_like(self.X)
            
        lr = self.initial_lr

        # Split the data onto the machines
        indices, num_samples_per_machine = self.__split_data(y)

        diff_losses = np.inf
        curr_loss = np.inf

        compute_loss_every = int(num_samples_per_machine / LOSS_PER_EPOCH)
        all_losses = np.zeros(int(num_samples_per_machine * self.num_epoch / compute_loss_every) + 1)

        train_start = time.time()
        np.random.seed(self.random_seed)

        # Decentralized SGD
        for epoch in range(0, self.num_epoch):
            if diff_losses > self.tol:
                for iteration in range(num_samples_per_machine):

                    curr_iteration = epoch * num_samples_per_machine + iteration

                    # Select a random sample for each machine
                    sample_indices = [np.random.choice(indices[machine]) for machine in range(0, n_machines)]

                    # Gradient step
                    self.X -= lr * self.gradient(A, y, sample_indices)

                    # Communication step
                    self.X = self.communicator.communicate(self.X, self.X_hat)

                    # Quantization step
                    self.X_hat += self.quantizer.quantize(self.X - self.X_hat)

                    # Compute the new loss
                    new_loss = self.loss(A, y)

                    # Compute the difference between last loss and current (check tol)
                    diff_losses = np.abs(new_loss - curr_loss)

                    # Update current loss value to new one
                    curr_loss = new_loss

                    # Update learning rate
                    lr = self.__update_lr(curr_iteration)

                    # Print iteration information
                    if curr_iteration % compute_loss_every == 0:
                        print('current iteration: {} epoch: {} epoch_iteration {} loss {} elapsed {}s'.format(curr_iteration,
                            epoch, iteration, curr_loss, time.time() - train_start))
                        all_losses[curr_iteration // compute_loss_every] = curr_loss

                    # If loss is infinite or NaN, stop the training
                    if np.isinf(curr_loss) or np.isnan(curr_loss):
                        print("Training interrupted, loss is either infinity or NaN")
                        break

            # Print epoch information
            print("epoch {}: loss {} score {}".format(epoch, curr_loss, self.score(A, y)))

        print("Training took: {}s".format(time.time() - train_start))

        return losses, all_losses
