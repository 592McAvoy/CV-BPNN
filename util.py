import numpy as np


class BN:
    def __init__(self):
        self.cache = None

    def forward(self, x, eps=1e-4):
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)
        out = x_hat
        self.cache = (x, x_hat, sample_mean, sample_var, eps)
        return out

    def backward(self, dout):
        x, x_hat, sample_mean, sample_var, eps = self.cache
        N = x.shape[0]

        dx_hat = dout
        dsigma = -0.5 * np.sum(dx_hat * (x - sample_mean), axis=0) * np.power(sample_var + eps, -1.5)
        dmu = -np.sum(dx_hat / np.sqrt(sample_var + eps), axis=0) - 2 * dsigma * np.sum(x - sample_mean, axis=0) / N
        dx = dx_hat / np.sqrt(sample_var + eps) + 2.0 * dsigma * (x - sample_mean) / N + dmu / N
        return dx


class Activate:
    def __init__(self):
        self.cache = None

    def sigmoidFoward(self, x):
        out = 1.0 / (1 + np.exp(-x))
        self.cache = out
        return out

    def sigmoidBackward(self, dout):
        out = self.cache
        return out*(1-out)*dout

    def tanhFoward(self, x):
        return 2 * self.sigmoidFoward(2 * x) - 1

    def tanhBackward(self, dout):
        out = self.cache
        return (1+out)*(1-out)*dout

    def ReLUFoward(self, x):
        out = np.maximum(x, 0)
        self.cache = x
        return out

    def ReLUBackward(self, dout):
        x = self.cache
        dtmp = np.zeros_like(dout)
        A, B = dout.shape
        for i in range(A):
            for j in range(B):
                if x[i, j] < 0:
                    dtmp[i, j] = 0
                else:
                    dtmp[i, j] = dout[i, j]
        return dtmp


