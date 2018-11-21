import numpy as np

class LDA:
    """linear discriminant analysis
    """
    def train(self, x, y):
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

    def predict(self, x):
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
    """Linear discriminant analysis, where covariance matrix is transformed to identity,
       then the classification is done by first transforming the input and second
       finding the nearest centroid
    """
    def train(self, x, y):
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
        self.mu = np.zoros((nc, n))
        for i, c in enumerate(self.cls):
            self.mu[i, :] = np.mean(x[y == c], axis=0)
            self.log_prob[i] = np.log(np.sum(y == c) / m)
        x_centred = x - self.mu[y]
        _, s, vt = np.linalg.svd(x_centred)
        self.trans = (1 / s) * vt
        self.mu_trans = np.dot(self.mu, self.trans.T)

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m x 1
        """
        m, n = x.shape
        nc = len(self.cls)
        diff = np.dot(x, self.trans.T).reshape((m, 1, n)) - self.mu_trans.reshape((1, nc, n))
        return self.cls[np.argmax(- m / 2 * np.sum(diff * diff, axis=2) + self.log_prob.reshape((1, nc)), axis=1)]

class FDA(LDA_SVD):
    """Fisher discriminant analysis
    """
    def train(self, x, y, p):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[Eq a], m x 1
            p: number of components
        outputs:
        modifies:
            self.mu_trans: array[[float]], nc x p
            self.cls: list[Eq a], nc x 1
            self.log_prob: array[float], nc x 1: log prob of class prior
            self.trans: array[[float]], p x n, operator that transforms data
        """
        super.train(x, y)
        self.p = p
        nc = len(self.cls)
        m, n = x.shape
        M = self.mu[y]
        S_w_inv = np.dot(self.trans.T, self.trans)
        MS_w_inv = np.dot(M, S_w_inv)
        tosvd = MS_w_inv - np.mean(MS_w_inv, axis=0)
        _, s, vt = np.linalg.svd(tosvd)
        proj = vt[:p,:]
        self.mu_trans_proj = np.dot(self.mu_trans, proj.T)
        self.trans_proj = np.dot(proj, self.trans)

    def predict(self, x):
        m, n = x.shape
        nc = len(self.cls)
        diff = np.dot(x, self.trans_proj.T).reshape((m, 1, p)) - self.mu_trans_proj.reshape((1, nc, p))
        return self.cls[np.argmax(- m / 2 * np.sum(diff * diff, axis=2) + self.log_prob.reshape((1, nc)), axis=1)]
