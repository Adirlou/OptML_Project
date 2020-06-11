# Optimization for Machine Learning: Mini-Project

## Convergence of Decentralized SGD under Various Topologies
*by Paul Griesser, Adrien Vandenbroucque, Robin Zbinden*

In this project, we propose Python code to compute a decentralized version of the SGD algorithm presented initially in the Github repository [here](https://github.com/epfml/ChocoSGD).

The contribution is twofold:

1) The original code from https://github.com/epfml/ChocoSGD has been partially rewritten in order to allow simulations with more machines and proposes more already implemented topologies. The user can also provide his own custom topology. In some cases, the execution speed is improved by computing the gradient in matrix form for all machines.

2) We created various experiences in order to try to answer the following questions:
    - We reproduce results which are well known and are the answer to the question: How does the network topology and number of nodes affect the convergence rate of decentralized SGD? What happens when one tries to optimize using a real-world network?
    - In order for the decentralized SGD to converge nicely in few iterations, the nodes in the network must be well connected. If one takes a graph such as the barbell graph or the path graph, does one notice an "information bottleneck"? That is, if there are very sparse cuts, does the information flows well in the network and does it affect the convergence rate of decentralized SGD?
    - One of the assumption of the paper about Choco-SGD (see https://arxiv.org/abs/1902.00340) is that the transition matrix is symmetric, which thus guarantees that the limiting distribution of the induced Markov Chain is uniform among the nodes. What happens when one allow for more general matrices? Are convergence results still obtained?

### How to reproduce the results?

All results displayed in the report can be obtained by running the various Jupyter Notebooks. In each of them, the user only has to run the cells, random seeds are already set. Here is a more detailed explanation for each notebook:

- `basic_topologies.ipynb`
    
    This Jupyter Notebook contains the experiments which aim at reproducing what was done in the Choco-SGD paper, but with a larger number of node. The goal is to see that already in these simple setups, we can see that the number of nodes affects the convergence, but most importantly the topology used can lead to very different results.
 
 - `real_network_topology.ipynb`
 
    This Jupyter Notebook contains the experiment which aims at computing the convergence behavior for a "real" network, and compare it to the "best case" which is when nodes are fully connected.
 
 - `bottleneck_topologies.ipynb`
 
    This Jupyter Notebook contains the experiments which aim at better understanding how topology affects convergence. In particular, we try to understand if sparse cuts in the underlying graph affect how the information flows between nodes during the optimization process.
    
 - `general_matrices.ipynb`
 
    This Jupyter Notebook contains the experiments which aim at generalizing some results obtained in the Choco-SGD paper. In particular, we perform the decentralized optimization on some specific ring topologies which are such that the transition matrix corresponding to the Markov chain is not symmetric.

### How to use the code?

The two instantiable classes for now are `DecentralizedSGDLogistic` (used to perform the decentralized SGD with logistic loss function) and `DecentralizedSGDLeastSquares`(used to perform the decentralized SGD with mean-square loss function). 
In order to fit the model (with logistic loss) with training dataset `A` and corresponding labels `y`, the user can use the following syntax:

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

    The stopping criterion. The training will stop when (|curr_loss - prev_loss| < tol). 
    
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
