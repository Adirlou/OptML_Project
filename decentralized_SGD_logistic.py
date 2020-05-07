import numpy as np
from scipy.special import expit as sigmoid

from decentralized_SGD_classifier import DecentralizedSGDClassifier


class DecentralizedSGDLogistic(DecentralizedSGDClassifier):
    """Class that encapsulates all attributes and methods needed to perform the decentralized SGD with a
    logistic regression classifier"""

    def loss(self, A, y):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x_mean_machines = self.X.mean(axis=1)
        pred = self.predict_proba(A)

        loss = (-1 / A.shape[0]) * (y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred)))

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

        # Need to densify matrix so that einsum can be used
        # Is fast since number of machines is usually not that high
        A_rand = np.asarray(A[sample_indices, :].T.todense())
        y_rand = y[sample_indices]

        # Column-wise dot product of all machines weights with their selected random sample,
        # on which we take the sigmoid to get the prediction for each machine
        predictions = sigmoid(np.einsum('kl,kl->l', self.X, A_rand))
        grad_matrix = (predictions - y_rand) * A_rand

        if self.regularizer:
            grad_matrix += self.regularizer * self.X

        return grad_matrix

    def predict(self, A):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        logits = A @ x
        pred = 1 * (logits >= 0.)
        return pred

    def predict_proba(self, A):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        logits = A @ x
        return sigmoid(logits)

    def score(self, A, y):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        pred = self.predict(A)
        acc = np.mean(pred == y)
        return acc
