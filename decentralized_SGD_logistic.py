import numpy as np
from scipy.special import expit as sigmoid

from decentralized_SGD_classifier import DecentralizedSGDClassifier


class DecentralizedSGDLogistic(DecentralizedSGDClassifier):
    """Class that encapsulates all attributes and methods needed to perform the decentralized SGD with a
    logistic regression classifier"""

    def loss(self, A, y):
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        loss = np.sum(np.log(1 + np.exp(-y * (A @ x)))) / A.shape[0]
        if self.regularizer:
            loss += self.regularizer * np.square(x).sum() / 2
        return loss
    
    def gradient(self, A, y, sample_idx, machine):

        x = self.X[:, machine]
        a = A[sample_idx]
        grad = -y[sample_idx] * a * sigmoid(-y[sample_idx] * a.dot(x).squeeze())

        # if isspmatrix(a):
            # minus_grad = minus_grad.toarray().squeeze(0) # Do we keep it ? TODO
        if self.regularizer:
            grad += self.regularizer * x
              
        return grad

    def predict(self, A):
        if not self.is_fitted:
            raise Exception("The model is unfitted")
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        logits = A @ x
        pred = 1 * (logits >= 0.)
        return pred
    
    def predict_proba(self, A):
        if not self.is_fitted:
            raise Exception("The model is unfitted")
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        logits = A @ x
        return sigmoid(logits)
    
    def score(self, A, y):
        if not self.is_fitted:
            raise Exception("The model is unfitted")
        # x = self.x_estimate if self.x_estimate is not None else self.x
        x = np.mean(self.X, axis=1)
        logits = A @ x
        pred = 2 * (logits >= 0.) - 1
        acc = np.mean(pred == y)
        return acc
