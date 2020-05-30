# Optimization for Machine Learning: Mini-Project

## Decentralized SGD using Choco-SGD
*by Paul Griesser, Adrien Vandenbroucque, Robin Zbinden*

In this project, we propose a Python code to compute a decentralized version of the SGD algorithm, initially presented [here](https://github.com/epfml/ChocoSGD).

The contribution is twofold:

1) The original code from https://github.com/epfml/ChocoSGD has been partially rewritten in order to allow simulations with more machines and proposes more already implemented topologies. The user can provide his own custom topology. In some cases, the execution speed is improved by computing the gradient in matrix form for all machines.

2) We created a few experiences in order to try to answer the following questions:
    - We reproduce results which are well known and are the answer to the question: How does the network topology and number of nodes affect the convergence rate of decentralized SGD? What happens when one tries to optimize using a real-world network?
    - In order for the decentralized SGD to converge nicely in few iterations, the nodes in the network must be well connected. If one takes a graph such as the barbell graph or the path graph, does one notice an "information bottleneck"? That is, if there are very sparse cuts, does the information flow well in the network and does it affect the convergence rate of decentalized SGD?
    - One of the assumption of the paper about Choco-SGD (see https://arxiv.org/abs/1902.00340) is that the transition matrix is symmetric, which thus guarantees that the limiting distribution of the induced Markov Chain is uniform among the nodes. What happens when one allow for more general matrices? Are convergence results still obtained?

### How to reproduce the results?

All results displayed in the report can be obtained by running the various Jupyter Notebooks. In each of them, the user only has to run the cells, the random seeds are already set.

### How to use the code?

The only instantiable class for now is `DecentralizedSGDLogistic`, which as its name indicates, is used to perform the decentralized SGD with logistic loss function. 
In order to fit the model with training dataset `A` and corresponding labels `y`, the user can use the following syntax:

```python
model = DecentralizedSGDLogistic()
model.fit(A, y)
```

Of course this version of the model uses all default values of the possible parameters. The parameters the user can input to this class are the following:

- **num_epoch:** ***int, default=1***

    The number of epoch for which to run the decentralized SGD.
    
- **initial_lr:** ***float, default=0.1***

    The initial learning rate.
    
- **lr_type:** ***{‘constant’, ‘decay’, ‘bottou’}, default='bottou'***

    The learning rate update rule.
    
    
- **tau:** ***float, default=None***
    
    Factor used when updating learning rate with `decay` rule.
    
- **tol:** ***float, default=0.001***

    The stopping criterion. If it is not None, training will stop when (loss > best_loss - tol) for n_iter_no_change consecutive epochs. 
    
- **regularizer:** ***float, default=0.0001***

    The constant that multiplies the regularization term.
    
- **n_machines:** ***int, default=1***

    The number of machines in the network.
    
- **topology:** ***{'disconnected', 'complete', 'ring', 'path', 'star', 'barbell', 'torus'} or np.ndarray, default='complete'***

    The topology of the network.
    
- **communication_method:** ***{'plain', 'choco'}, default='plain'***

    The method to use when machine communicate with their neighbors.
    
- **consensus_lr:** ***float, default=1.0***

    The consensus learning rate, used when the communication method is set to *choco*.
    
- **communication_frequency:** ***int, default=1***

    The frequency at which the communication is done between machines. For example, if it set to 2, then the machines will only communicate every two iterations.
    
- **data_distribution_strategy:** ***{'undistributed', 'naive', 'random', 'label-sorted'}, default='naive'***

    The strategy to use for distributing the data across the machines. If set to `None`, each machine will have access to the entire dataset. 
    
- **data_distribution_random_seed:** ***int, default=None***
    
    The seed to use when data is distributed randomly across the machines.
    
- **quantization_method:** ***{'full', 'top', 'random-biased', 'random-unbiased'}, default="full"***

    The quantization method to use.
    
- **features_to_keep:** ***int, default=None***

    The number of features to keep when quantization is done with methods *top*, *random-biased*, or *random-unbiased*.
    
- **random_seed:** ***int, default=None***

    The random seed to be used when fitting the model.
    
- **compute_loss_every:** ***int, default=50***

    The frequency at which the loss value is recorded. For example, if it set to 2, then the loss value will be recorded every two iterations.
