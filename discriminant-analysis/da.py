import numpy as np

class LDA:
    def train(x, y):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[Eq a], m x 1
        outputs:
        modifies:
            self.mu: array[[float]], nc x n
            self.sigma: array[[float]], n x n: covariance matrix
            self.sigma_inv: array[[float]], n x n: inverse of covariance matrix
            self.cls: list[Eq a], nc x 1
            self.log_prob: array[float], nc x 1: log prob of class prior
        """
        self.cls = list(set(y))
        nc = len(self.cls)
        m, n = x.shape
        self.mu = np.zoros((nc, n))
        self.sigma = np.zeros((n, n))
        for i, c in enumerate(self.cls):
            self.mu[i, :] = np.mean(x[y == c], axis=0)
            self.log_prob[i] = np.log(np.sum(y == c)) - np.log(m)
        self.sigma = np.dot(x.T, x) / m
        self.sigma_inv = np.linalg.inv(sigma)

    def predict(x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m x 1
        """
        m, n = x.shape
        nc = len(self.cls)
        dot_prod = np.zeros((nc, m))
        for i in range(nc):
            x_shifted = x - self.mu[i]
            dot_prod[i, :] = np.sum(np.dot(x_shifted, self.sigma_inv) * x_shifted, axis=1) - log_prob[i]
        return np.argmin(dot_prod, axis=0)

class LDA_SVD:
    def train(x, y):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[Eq a], m x 1
        outputs:
        modifies:
            self.mu_trans: array[[float]], nc x n
            self.cls: list[Eq a], nc x 1
            self.log_prob: array[float], nc x 1: log prob of class prior
            self.trans: array[[float]], n x n, operator that transforms data to standard normal
        """
        self.cls = list(set(y))
        nc = len(self.cls)
        m, n = x.shape
        mu = np.zoros((nc, n))
        for i, c in enumerate(self.cls):
            mu[i, :] = np.mean(x[y == c], axis=0)
            self.log_prob[i] = np.log(np.sum(y == c)) - np.log(m)
        x_centred = x - self.mu[y]
        _, s, vt = np.linalg.svd(x_centred)
        self.trans = (1 / s) * vt
        self.mu_trans = np.dot(self.trans, mu.T).T

    def predict(x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m x 1
        """
        m, n = x.shape
        nc = len(self.cls)
        diff = x.reshape((m, 1, n)) - self.mu_trans.reshape((1, nc, n))
        return np.argmax(np.sum(diff * diff, axis=2) - self.log_prob.reshape((1, nc)), axis=1)
