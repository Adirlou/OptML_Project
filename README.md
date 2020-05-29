# Optimization for Machine Learning: Mini-Project

## Decentralized SGD using Choco-SGD
*by Paul Griesser, Adrien Vandenbroucque, Robin Zbinden*

In this project, we propose a Python code to compute a decentralized version of the SGD algorithm, presented here https://github.com/epfml/ChocoSGD.

The contribution is twofold:

1) The original code from https://github.com/epfml/ChocoSGD has been rewritten partially in order to allow simulations with more machines. In some cases, the execution speed is also improved.

2) We created a few experiences in order to try to answer the following questions:
    - We reproduce results which are well known and are the answer to the question: How does the network topology and number of nodes affect the convergence rate of decentralized SGD? What happens when one tries to optimize using a real-world network?
    - In order for the decentralized SGD to converge nicely in few iterations, the nodes in the network must be well connected. If one takes a graph such as the barbell graph or the path graph, does one notice an "information bottleneck"? That is, if there are very sparse cuts, does the information flow well in the network and does it affect the convergence rate of decentalized SGD?
    - One of the assumption of the paper about Choco-SGD (see https://arxiv.org/abs/1902.00340) is that the transition matrix is symmetric, which thus guarantees that the limiting distribution of the induced Markov Chain is uniform among the nodes. What happens when one allow for more general matrices? Are convergence results still obtained?
    
### How to use the code?

TODO

### How to reproduce the results?

TODO
