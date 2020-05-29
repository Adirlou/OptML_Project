import numpy as np


class Quantizer:
    """
    Class representing the sparsification (quantization) method
    to reduce the transmitted number of bit during the communication step of DSGD.
    """

    def __init__(self, method, features_to_keep=None):
        # k: number of features to keep (usefull only for "full", "random-biased",
        # and "random-unibiased" quantizer)
        self.method = method
        self.features_to_keep = features_to_keep

        self.__validate_params()

    def __validate_params(self):
        """Validate input parameters."""

        top_method = ['top']
        full_method = ['full']
        random_methods = ['random-biased', 'random-unbiased',]
        valid_methods = top_method + full_method + random_methods

        # Check if method is valid
        if self.method not in valid_methods:
            raise ValueError('Method for quantization should be one of: ' + str(valid_methods))

        # If using random methods or "top", need to set the number of features to keep properly
        if self.method in top_method + random_methods:
            if not self.features_to_keep:
                raise ValueError('Parameter "features_to_keep" must be set to use with methods ' + str(top_method + random_methods))

            # Check if number of machines is an integer
            if not isinstance(self.features_to_keep, int):
                raise ValueError('Invalid parameter "features_to_keep", must be an integer')

            # Check if number of machines is a positive integer
            if self.features_to_keep <= 0:
                raise ValueError('Parameter "features_to_keep" must be set to use with methods ' + str(top_method + random_methods))


    def quantize(self, weight_matrix):
        """Perform the quantization step of the decentralized SGD,
        depending on the given method"""
        if self.method == 'full':
            return weight_matrix
        elif self.method == 'top':
            return self.__quantize_top(weight_matrix)
        elif self.method in ['random-biased', 'random-unbiased']:
            return self.__quantize_random(weight_matrix)

    def __quantize_top(self, weight_matrix):
        # keep only k highest features (other features are set to zero)

        n_features, n_machines = weight_matrix.shape

        if self.features_to_keep >= n_features:
            return weight_matrix
        else:
            quantized = np.zeros_like(weight_matrix)

            for i in range(0, n_machines):
                indexes = np.argsort(-np.abs(weight_matrix[:, i]))
                quantized[indexes[:self.features_to_keep], i] = weight_matrix[indexes[:self.features_to_keep], i]
            return quantized

    def __quantize_random(self, weight_matrix):
        # randomly choose k features to keep (other features are set to zero)

        n_features, n_machines = weight_matrix.shape

        if self.features_to_keep >= n_features:
            return weight_matrix
        else:
            quantized = np.zeros_like(weight_matrix)

            for i in range(0, n_machines):
                indexes = np.random.choice(np.arange(n_features), k, replace=False)
                quantized[indexes[:self.features_to_keep], i] = weight_matrix[indexes[:self.features_to_keep], i]

            if self.method == 'random-unbiased':
                # normalize the kept features so that the quantized gradient is unbiased w.r.t.
                # the real gradient
                return weight_matrix.shape[0] / k * quantized
            return quantized
