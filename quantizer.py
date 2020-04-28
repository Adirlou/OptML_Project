import numpy as np


class Quantizer:
    """
    Class representing the sparsification (quantization) method
    to reduce the transmitted number of bit during the communication step of DSGD.
    """

    def __init__(self, quantization, k=None, s=None):
        # k: number of features to keep (usefull only for "full", "random-biased",
        # and "random-unibiased" quantizer)
        # s: number of quantization levels (usefull only for 'qsgd-biased'
        # and 'qsgd-unbiased' quantizer)

        assert quantization in ['full', 'top', 'random-biased', 'random-unbiased',
                                'qsgd-biased', 'qsgd-unbiased']
        if quantization in ['top', 'random-biased', 'random-unbiased']:
            assert (k is not None) and (k > 0)
            self.features_to_keep = k
        if quantization in ['qsgd-biased', 'qsgd-unbiased']:
            assert (s is not None) and (s >= 1)
            self.num_levels = s
        self.quantization = quantization

    def quantize(self, x):
        # quantize according to quantization function
        # x: shape(num_features, n_cores) contains the stochastic gradient vectors for each machine
        n_cores = x.shape[1]
        n_features = x.shape[0]

        if self.quantization == 'full':
            # do not sparsify gradient
            return x

        if self.quantization == 'top':
            # keep only k highest features (other features are set to zero)
            assert k <= n_features
            q = np.zeros_like(x)
            k = self.features_to_keep
            for i in range(0, n_cores):
                indexes = np.argsort(np.abs(x[:, i]))[::-1]
                q[indexes[:k], i] = x[indexes[:k], i]
            return q

        if self.quantization in ['random-biased', 'random-unbiased']:
            # randomly choose k features to keep (other features are set to zero)
            assert k <= n_features
            q = np.zeros_like(x)
            k = self.features_to_keep
            for i in range(0, n_cores):
                indexes = np.random.choice(np.arange(n_features), k, replace=False)
                q[indexes[:k], i] = x[indexes[:k], i]

            if self.quantization == 'random-unbiased':
                # normalize the kept features so that the quantized gradient is unbiased w.r.t.
                # the real gradient
                return x.shape[0] / k * q
            return q

        if self.quantization in ['qsgd-biased', 'qsgd-unbiased']:
            # quantize gradients using gsgd function
            is_biased = (self.quantization == 'qsgd-biased')
            q = np.zeros_like(x)
            for i in range(0, n_cores):
                q[:, i] = self.qsgd_quantize(x[:, i], self.num_levels, is_biased)
            return q

    def qsgd_quantize(self, x, s, is_biased):
        # gsgd quantization function c.f. https://arxiv.org/pdf/1610.02132.pdf page 5
        norm = np.linalg.norm(x)
        level_float = s * np.abs(x) / norm
        previous_level = np.floor(level_float)
        is_next_level = np.random.rand(*x.shape) < (level_float - previous_level)
        new_level = previous_level + is_next_level
        scale = 1
        if is_biased:
            n_features = x.shape[0]
            scale = 1. / (np.minimum(n_features / s ** 2, np.sqrt(n_features) / s) + 1.)
        return scale * np.sign(x) * norm * new_level / s
