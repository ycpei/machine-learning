"""
Copyright Yuchen Pei (2018) (hi@ypei.me), licensed under GNU GPLv3+
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
            self.mu: array[[float]], nc x n, per class mean
            self.M: array[[float]], m x n, per sample mean
            self.S_w_inv: array[[float]], n x n: inverse of covariance matrix
            self.cls: array[Eq a], nc: classes
            self.log_prob: array[float], nc: log prob of class prior
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
        self.M = self.mu[y_idx]
        self.x_centred = (x - self.M) / np.sqrt(m - nc)
        S_w = np.dot(self.x_centred.T, self.x_centred)
        if np.isclose(np.linalg.det(S_w), 0) and type(self) == LDA:
            raise ZeroDivisionError('Low rank covariance matrix, please use LDA_SVD instead.')
        self.S_w_inv = np.linalg.inv(S_w)

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
            y: array[Eq a], m
        outputs:
        modifies:
            as in super.train
            self.trans: array[[float]], n x n, operator that transforms data to standard normal
            self.mu_trans: array[[float]], nc x n
        """
        super().train(x, y)
        nc = len(self.cls)
        m, n = x.shape
        _, s, vt = np.linalg.svd(self.x_centred)
        vt = vt[np.logical_not(np.isclose(s, 0))]
        s = s[np.logical_not(np.isclose(s, 0))]
        self.trans = (vt.T / s).T
        self.mu_trans = np.dot(self.mu, self.trans.T)

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m
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
            p: modified as a local var
            self.xbar: array[float], n, mean of input x
            self.trans_proj: array[[float]], p x n, operator that first transforms data to standard normal,
                then projects them to the principle component space
            self.coef: array[[float]], nc x n, coefficient in the linear model
            self.intercept: array[float], nc, intercept in the linear model
        """
        super().train(x, y)
        nc = len(self.cls)
        if p is None:
            p = nc - 1
        m, n = x.shape
        mu_trans = np.dot(self.mu, self.trans.T)
        self.xbar = np.mean(x, axis=0)
        xbar_trans = np.sum(mu_trans * np.exp(self.log_prob).reshape(-1, 1), axis=0)
        M_trans = np.dot(self.M, self.trans.T)
        tosvd = M_trans - xbar_trans
        _, s, vt = np.linalg.svd(tosvd)
        vt = vt[np.logical_not(np.isclose(s, 0))]
        if p > len(vt):
            warnings.warn('Rank is lower than the number of components, use rank as number of components')
            p = len(vt)
        proj = vt[:p,:]
        self.trans_proj = np.dot(proj, self.trans)
        self.coef = np.dot(self.mu, np.dot(self.trans_proj.T, self.trans_proj))
        self.intercept = - .5 * np.sum(np.dot(self.mu, self.trans_proj.T) ** 2, axis=1) + self.log_prob
        self.intercept.shape = -1, 1

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[Eq a], m
        """
        return self.cls[np.argmax(np.dot(self.coef, x.T) + self.intercept, axis=0)]

    def transform_to_match_sk(self, x):
        """transformation of x, to match the output of sklearn.
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[[float]], m x n
        """
        return np.dot(x - self.xbar, self.trans_proj.T)

class QDA:
    def train(self, x, y):
        """train
        inputs:
            x: array[[float]], m x n
            y: array[Eq], m
        outputs:
        modifies:
            self.cls: array[Eq a], nc: classes
            self.A: array[[[float]]], nc x n x n: quadratic term matrix
            self.B: array[[float]], nc x n: linear term matrix
            self.C: array[float], nc: constant term
        """
        self.cls = np.array(list(set(y)))
        nc = len(self.cls)
        m, n = x.shape
        self.A = np.zeros((nc, n, n))
        self.B = np.zeros((nc, n))
        self.C = np.zeros(nc)
        for i, c in enumerate(self.cls):
            xc = x[y == c]
            mu = np.mean(xc, axis=0)
            mc = xc.shape[0]
            if mc == 1:
                raise ZeroDivisionError("Only one sample in class {}".format(c))
            else:
                x_centred = (xc - mu) / np.sqrt(mc - 1)
            _, s, vt = np.linalg.svd(x_centred)
            if np.sum(np.isclose(s, 0)) > 0:
                raise ZeroDivisionError("Singular covariance matrix in class {}".format(c))
            trans = vt / s.reshape(-1, 1) #shape = r, n
            self.A[i, :] = -.5 * np.dot(trans.T, trans) #shape = n, n
            self.B[i, :] = -2 * np.dot(mu, self.A[i, :]) #shape = n,
            self.C[i] = -.5 * np.sum(self.B[i, :] * mu) \
                        - np.sum(np.log(s)) + np.log(mc / m)

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            array[Eq], m: prediction labels
        """
        quad = np.sum(np.transpose(np.dot(x, self.A), (1, 0, 2)) * x, axis=2) #shape = nc, m
        lin = np.dot(self.B, x.T) #shape = nc, m
        return self.cls[np.argmax(quad + lin + self.C.reshape(-1, 1), axis=0)]

    def decision_function_sk_debug(self, x):
        """decision function for debugging purposes, the output should match that of sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis._decision_function
        """
        quad = np.sum(np.transpose(np.dot(x, self.A), (1, 0, 2)) * x, axis=2) #shape = nc, m
        lin = np.dot(self.B, x.T) #shape = nc, m
        return (quad + lin + self.C.reshape(-1, 1)).T
