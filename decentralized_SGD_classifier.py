import time
import numpy as np

INIT_WEIGHT_STD = 1
LOSS_PER_EPOCH = 10

class DecentralizedSGDClassifier():
    def __init__(self, num_epoch,
                 lr_type,
                 initial_lr=None,
                 regularizer=None,
                 epoch_decay_lr=None,
                 consensus_lr=None,
                 quantization="full",
                 # number of coordinates k in top-k or random-k quantization
                 coordinates_to_keep=None,
                 # number of levels in qsgd quantization
                 num_levels=None,
                 estimate='final',
                 # number of machines
                 n_cores=1,
                 topology='centralized',
                 method='choco',
                 distribute_data=False,
                 # whether each machine gets random data or continuous set of data
                 # might not have any difference, depends on the dataset
                 split_data_strategy=None,
                 tau=None,
                 real_update_every=1,
                 random_seed=None,
                 split_data_random_seed=None,
                 ):
        
        # Assertion on the parameters 
        if lr_type in ['constant', 'decay']:
            assert initial_lr > 0
        if lr_type == 'decay':
            assert initial_lr and tau
            assert regularizer > 0
        if lr_type == 'epoch-decay':
            assert epoch_decay_lr is not None

        assert estimate in ['final', 'mean', 't+tau', '(t+tau)^2'] # Not used yet TODO

        assert method in ['choco', 'dcd-psgd', 'ecd-psgd', 'plain']
        if method in ['dcd-psgd', 'ecd-psgd']:
            assert quantization in ['random-unbiased', 'qsgd-unbiased']

        if not distribute_data:
            assert not split_data_strategy
        else:
            assert split_data_strategy in ['naive', 'random', 'label-sorted']

        self.num_epoch = num_epoch
        self.lr_type = lr_type
        self.initial_lr = initial_lr
        self.regularizer = regularizer # Maybe we can move it to children classes TODO
        self.epoch_decay_lr = epoch_decay_lr
        self.coordinates_to_keep = coordinates_to_keep
        self.estimate = estimate # Not used yet TODO
        self.tau = tau
        self.real_update_every = real_update_every
        self.random_seed = random_seed
        self.method = method
        self.distribute_data = distribute_data
        self.split_data_strategy = split_data_strategy
        self.split_data_random_seed = split_data_random_seed
        
        # self.quantizer = Quantization(quantization, coordinates_to_keep, num_levels) TODO
        # self.communicator = Communicator(consensus_lr, n_cores, topology, method) TODO
        
        self.X = None
        self.is_fitted = False

        
    """
    Compute the learning rate at the given epoch, iteration
    """
    def __lr(self, epoch, iteration, num_samples, d):
        t = epoch * num_samples + iteration
        if self.lr_type == 'constant':
            return self.initial_lr
        if self.lr_type == 'epoch-decay':
            return self.initial_lr * (self.epoch_decay_lr ** epoch)
        if self.lr_type == 'decay':
            return self.initial_lr / (self.regularizer * (t + self.tau))
        if self.lr_type == 'bottou':
            return self.initial_lr / (1 + self.initial_lr * self.regularizer * t)
     
    """
    Split the data onto machines following the split data strategy
    """
    def __split_data(self, num_samples, n_cores):
        
        if self.distribute_data:
            np.random.seed(self.split_data_random_seed)
            num_samples_per_machine = num_samples // n_cores
            if self.split_data_strategy == 'random':
                all_indexes = np.arange(num_samples)
                np.random.shuffle(all_indexes)
            elif split_data_strategy == 'naive':
                all_indexes = np.arange(num_samples)
            elif split_data_strategy == 'label-sorted':
                all_indexes = np.argsort(y)

            indices = []
            for machine in range(0, n_cores - 1):
                indices += [all_indexes[num_samples_per_machine * machine:\
                                        num_samples_per_machine * (machine + 1)]]
            indices += [all_indexes[num_samples_per_machine * (n_cores - 1):]]
            print("length of indices:", len(indices))
            print("length of last machine indices:", len(indices[-1]))
        else:
            num_samples_per_machine = num_samples
            indices = np.tile(np.arange(num_samples), (n_cores, 1)) 
            
        return indices, num_samples_per_machine
    
     
    def loss(self, A, y):
        raise NotImplementedError("Abstract method")
    
    def gradient(self, A, y, sample_idx, machine=None):
        raise NotImplementedError("Abstract method")
        
    def predict(self, A):
        raise NotImplementedError("Abstract method")
        
    def score(self, A, y):
        raise NotImplementedError("Abstract method")
        
        
    def fit(self, A, y_init):
        
        y = np.copy(y_init)
        num_samples, num_features = A.shape
        n_cores = quantizer.n_cores() # TODO
        losses = np.zeros(self.num_epoch + 1)
        
        # Initialization of parameters
        self.X = np.random.normal(0, INIT_WEIGHT_STD, size=(num_features,))
        self.X = np.tile(self.X, (n_cores, 1)).T
        X_hat = np.zeros((n_cores, num_features)) 
        
        # Split the data onto the machines
        indices, num_samples_per_machine = __split_data(num_samples, n_cores)
            
        # epoch 0 loss evaluation
        losses[0] = self.loss(A, y)

        compute_loss_every = int(num_samples_per_machine / LOSS_PER_EPOCH)
        all_losses = np.zeros(int(num_samples_per_machine * self.num_epoch / compute_loss_every) + 1)

        train_start = time.time()
        np.random.seed(self.random_seed)
            
        for epoch in np.arange(self.num_epoch):
            for iteration in range(num_samples_per_machine):
                
                t = epoch * num_samples_per_machine + iteration
                
                if t % compute_loss_every == 0:
                    loss = self.loss(A, y)
                    print('{} t {} epoch {} iter {} loss {} elapsed {}s'.format(p, t,
                        epoch, iteration, loss, time.time() - train_start))
                    all_losses[t // compute_loss_every] = loss
                    if np.isinf(loss) or np.isnan(loss):
                        print("finish trainig")
                        break
                
                lr = self.__lr(epoch, iteration, num_samples_per_machine, num_features)
                
                # Gradient step
                x_plus = np.zeros_like(self.x)
                for machine in range(0, n_cores):
                    sample_idx = np.random.choice(indices[machine])
                    
                    minus_grad = -1 * self.gradient(A, y, machine, sample_idx)
                    
                    x_plus[:, machine] = lr * minus_grad
                    
                    # Quantization step TODO
                    # Communication step TODO
                    
            losses[epoch + 1] = self.loss(A, y)
            print("epoch {}: loss {} score {}".format(epoch, losses[epoch + 1], self.score(A, y)))
            if np.isinf(losses[epoch + 1]) or np.isnan(losses[epoch + 1]):
                print("finish trainig")
                break

        print("Training took: {}s".format(time.time() - train_start))
        
        self.isFitted = True

        return losses, all_losses