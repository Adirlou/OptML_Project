# This code is meant to improve existing code which comes from
# https://github.com/epfml/ChocoSGD/tree/master/convex_code.
# As such, much of the code below is taken directly from that repository,
# and the credits should go to its authors.

import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import isspmatrix as is_sparse_matrix

from decentralized_SGD_classifier import DecentralizedSGDClassifier

class DecentralizedSGDLogistic(DecentralizedSGDClassifier):
    """Class that encapsulates all attributes and methods needed to perform the decentralized SGD with a
    logistic regression classifier"""

    def loss(self, A, y):
        """Compute the logistic loss given the input data and corresponding labels.
        :param A: input data
        :param y: target data
        """
        x_mean_machines = self.X.mean(axis=1)

        # Make sure that no pred is 0 or 1 otherwise log-loss is undefined
        pred = np.clip(self.predict_proba(A), 1e-15, 1- 1e-15)

        # Compute the loss
        loss = -(y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))) / A.shape[0]

        # Add regularization
        if self.regularizer:
            loss += self.regularizer * np.sum(x_mean_machines**2) / 2
        return loss

    def gradient_old(self, A, y, sample_idx, machine):

        x = self.X[:, machine]
        a = A[sample_idx]

        # Compute the gradient
        predicted = sigmoid(a.dot(x))
        grad =  (predicted - y[sample_idx]) * a

        if self.regularizer:
            grad += self.regularizer * x

        return grad

    def gradient(self, A, y, sample_indices):
        """Compute the logistic loss gradient of the weights of each machine
        w.r.t. the chosen random sample at each machine
        :param A: input data
        :param y: target data
        :param sample_indices: indices of the selected sample for each machine
        """
        # Get for each machine the corresponding chosen sample and make sure
        # that the shape is such that a column corresponds to a machine
        A_rand = A[sample_indices, :].T

        # If A is in a sparse format
        if is_sparse_matrix(A_rand):
            # Need to densify matrix so that einsum can be used
            # Is fast since number of machines is usually not that high
            A_rand = np.asarray(A_rand.todense())

        # Get the corresponding label for each chosen sample
        y_rand = y[sample_indices]

        # Column-wise dot product of all machines weights with their selected random sample,
        # on which we take the sigmoid to get the prediction for each machine
        predictions = sigmoid(np.einsum('kl,kl->l', self.X, A_rand))

        # Matrix whose columns contains the gradient corresponding
        # to each machine
        grad_matrix = (predictions - y_rand) * A_rand

        # Add regularizer if needed
        if self.regularizer:
            grad_matrix += self.regularizer * self.X

        return grad_matrix

    def predict(self, A):
        """Predict target data of input data A using the fitted model.
        :param A: input data
        """
        x_mean_machines = self.X.mean(axis=1)
        logits = A @ x_mean_machines
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        """Predict target probabilities data of input data A using the fitted
        model.
        :param A: input data
        """
        x_mean_machines = self.X.mean(axis=1)
        logits = A @ x_mean_machines
        return sigmoid(logits)
