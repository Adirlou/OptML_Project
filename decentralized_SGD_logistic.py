import numpy as np
from scipy.special import expit as sigmoid
from scipy.sparse import isspmatrix as is_sparse_matrix

from decentralized_SGD_classifier import DecentralizedSGDClassifier
from sklearn.metrics import log_loss

class DecentralizedSGDLogistic(DecentralizedSGDClassifier):
    """Class that encapsulates all attributes and methods needed to perform the decentralized SGD with a
    logistic regression classifier"""

    def loss(self, A, y):
        # x = self.x_estimate if self.x_estimate is not None else self.x
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
        A_rand = A[sample_indices, :].T

        if is_sparse_matrix(A_rand):
            # Need to densify matrix so that einsum can be used
            # Is fast since number of machines is usually not that high
            A_rand = np.asarray(A_rand).todense()

        y_rand = y[sample_indices]

        # Column-wise dot product of all machines weights with their selected random sample,
        # on which we take the sigmoid to get the prediction for each machine
        predictions = sigmoid(np.einsum('kl,kl->l', self.X, A_rand))

        # Matrix whose columns contains the gradient corresponding
        # to each machine
        grad_matrix = (predictions - y_rand) * A_rand

        if self.regularizer:
            grad_matrix += self.regularizer * self.X

        return grad_matrix

    def predict(self, A):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x_mean_machines = self.X.mean(axis=1)
        logits = A @ x_mean_machines
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x_mean_machines = self.X.mean(axis=1)
        logits = A @ x_mean_machines
        return sigmoid(logits)

    def score(self, A, y):
        pred = self.predict(A)
        acc = np.mean(pred == y)
        return acc
