"""
Copyright Yuchen Pei (2018) (hi@ypei.me), licensed under GPLv3+
"""

import numpy as np
import warnings

class LDA:
    """vanilla linear discriminant analysis
    """
    def train(self, x, y):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[Eq a], m x 1
        outputs:
        modifies:
            self.mu: array[[float]], nc x n
            self.S_w_inv: array[[float]], n x n: inverse of covariance matrix
            self.cls: array[Eq a], nc x 1
            self.log_prob: array[float], nc x 1: log prob of class prior
            self.x_centred: array[[float]]: m x n
        """
        self.cls = np.array(list(set(y)))
        nc = len(self.cls)
        cls_map = {c: i for i, c in enumerate(self.cls)}
        y_idx = np.array([cls_map[c] for c in y])
        m, n = x.shape
        self.mu = np.zeros((nc, n))
        self.log_prob = np.zeros(nc)
        for i, c in enumerate(self.cls):
            self.mu[i, :] = np.mean(x[y == c], axis=0)
            self.log_prob[i] = np.log(np.sum(y == c) / m)
        M = self.mu[y_idx]
        self.x_centred = (x - M) / np.sqrt(m - nc)
        S_w = np.dot(self.x_centred.T, self.x_centred)
        if np.isclose(np.linalg.det(S_w), 0) and type(self) == LDA:
            raise ZeroDivisionError('Low rank covariance matrix, please use LDA_SVD instead.')
        self.S_w_inv = np.linalg.inv(S_w)
        #print(S_b / m, np.linalg.inv(np.linalg.inv(S_b)) / m)
        #print(np.linalg.det(S_b))
        #print(np.linalg.svd(S_b))
        #print(np.dot(S_b, np.linalg.inv(S_b)))

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m x 1
        """
        m, n = x.shape
        nc = len(self.cls)
        p = np.zeros((nc, m))
        for i in range(nc):
            x_shifted = x - self.mu[i]
            p[i, :] = - .5 * np.sum(np.dot(x_shifted, self.S_w_inv) * x_shifted, axis=1) + self.log_prob[i]
        #print(p)
        return self.cls[np.argmax(p, axis=0)]

class LDA_SVD(LDA):
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
            as in super.train
            self.trans: array[[float]], n x n, operator that transforms data to standard normal
            self.mu_trans: array[[float]], nc x n
        """
        super().train(x, y)
        nc = len(self.cls)
        #print("mu", self.mu)
        #print("x", x[:10, :])
        #print("x_centred:", self.x_centred[:10,:])
        m, n = x.shape
        _, s, vt = np.linalg.svd(self.x_centred)
        vt = vt[np.logical_not(np.isclose(s, 0))]
        s = s[np.logical_not(np.isclose(s, 0))]
        self.trans = (1 / s.reshape(-1, 1)) * vt
        #print("da:", self.trans)
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
        diff = np.dot(x, self.trans.T).reshape((m, 1, -1)) - self.mu_trans.reshape((1, nc, -1))
        return self.cls[np.argmax(- .5 * np.sum(diff * diff, axis=2) + self.log_prob.reshape((1, nc)), axis=1)]

class FDA(LDA_SVD):
    """Fisher discriminant analysis
    """
    def train(self, x, y, p=None):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[Eq a], m x 1
            p: int, number of components
        outputs:
        modifies:
            as in super.train
            self.p: int
            self.trans_proj: array[[float]], p x n, operator that first transforms data to standard normal,
                then projects them to the principle component space
            self.mu_trans_proj: array[[float]], p x n, the centroids in the principle component space
        """
        super().train(x, y)
        nc = len(self.cls)
        if p is None:
            p = nc - 1
        m, n = x.shape
        mu_trans = np.dot(self.mu, self.trans.T)
        self.xbar = np.mean(x, axis=0)
        xbar_trans = np.sum(mu_trans * np.exp(self.log_prob).reshape(-1, 1), axis=0)
        tosvd = mu_trans - xbar_trans
        _, s, vt = np.linalg.svd(tosvd)
        vt = vt[:nc, :]
        vt = vt[np.logical_not(np.isclose(s, 0))]
        if p > len(vt):
            warnings.warn('Rank is lower than the number of components, use rank as number of components')
            p = len(vt)
        proj = vt[:p,:]
        self.p = p
        self.mu_trans_proj = np.dot(self.mu_trans, proj.T)
        self.trans_proj = np.dot(proj, self.trans)
        #print("da", self.trans)
        #print("da", proj)
        #print("da", self.trans_proj)

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m x 1
        """
        m, n = x.shape
        nc = len(self.cls)
        diff = np.dot(x, self.trans_proj.T).reshape((m, 1, self.p)) - self.mu_trans_proj.reshape((1, nc, self.p))
        return self.cls[np.argmax(- .5 * np.sum(diff * diff, axis=2) + self.log_prob.reshape((1, nc)), axis=1)]

    def transform_to_match_sk(self, x):
        """transformation of x, to match the output of sklearn. currently not matching except when doing dimensionality reduction (p < nc - 1).
        """
        return np.dot(x - self.xbar, self.trans_proj.T)
