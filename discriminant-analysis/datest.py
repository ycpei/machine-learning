from da import *
import unittest
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
#from da_sklearn import LinearDiscriminantAnalysis

class TestLDA(unittest.TestCase):
    def setUp(self):
        self.clf = LDA()
        self.clf_sk = LinearDiscriminantAnalysis(store_covariance=True)

    def assertAgainstSK(self, x, y, x_new):
        m, n = x.shape
        self.clf.train(x, y)
        self.clf_sk.fit(x, y)
        np.testing.assert_equal(self.clf.cls, self.clf_sk.classes_)
        # to test covariance: change m - nc to m in the code
        #np.testing.assert_almost_equal(np.linalg.inv(self.clf.S_w_inv), self.clf_sk.covariance_)
        np.testing.assert_almost_equal(np.exp(self.clf.log_prob), self.clf_sk.priors_)
        np.testing.assert_almost_equal(self.clf.predict(x), self.clf_sk.predict(x)) #
        np.testing.assert_almost_equal(self.clf.predict(x_new), self.clf_sk.predict(x_new))

    def testAgainstSkDocs(self):
        x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        x_new = np.array([[-0.8, -1]])
        self.assertAgainstSK(x, y, x_new)

    def testLowRank(self):
        # when m and n are close: mismatch covariances due to low rank (won't pass)
        np.random.seed(0)
        x = np.random.randn(6, 4)
        y = np.random.randint(0, 3, size=(6,))
        self.assertRaises(ZeroDivisionError, self.clf.train, x, y)

    def testAgainstSkRandom(self):
        #np.random.seed(6)
        x = np.random.randn(6, 2)
        y = np.random.randint(0, 2, size=(6,))
        x_new = np.random.randn(5, 2)
        self.assertAgainstSK(x, y, x_new)
        #np.random.seed(5)
        x = np.random.randn(100, 4)
        y = np.random.randint(0, 3, size=(100,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)

class TestLDA_SVD(unittest.TestCase):
    def setUp(self):
        self.clf = LDA_SVD()
        self.clf_sk = LinearDiscriminantAnalysis()

    def assertAgainstSK(self, x, y, x_new):
        m, n = x.shape
        self.clf.train(x, y)
        self.clf_sk.fit(x, y)
        np.testing.assert_equal(self.clf.cls, self.clf_sk.classes_)
        np.testing.assert_almost_equal(np.exp(self.clf.log_prob), self.clf_sk.priors_)
        np.testing.assert_almost_equal(self.clf.predict(x), self.clf_sk.predict(x)) #
        np.testing.assert_almost_equal(self.clf.predict(x_new), self.clf_sk.predict(x_new))

    def testAgainstSkDocs(self):
        x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        x_new = np.array([[-0.8, -1]])
        self.assertAgainstSK(x, y, x_new)

    def testLowRank(self):
        # when m and n are close: low rank covariance matrix
        np.random.seed(0)
        x = np.random.randn(6, 4)
        y = np.random.randint(0, 3, size=(6,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)

    def testAgainstSkRandom(self):
        #np.random.seed(6)
        x = np.random.randn(6, 2)
        y = np.random.randint(0, 2, size=(6,))
        x_new = np.random.randn(5, 2)
        self.assertAgainstSK(x, y, x_new)
        #np.random.seed(5)
        x = np.random.randn(100, 4)
        y = np.random.randint(0, 3, size=(100,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)

class TestFDA(unittest.TestCase):
    def setUp(self):
        self.clf = FDA()

    def assertAgainstSK(self, x, y, x_new, p=None):
        m, n = x.shape
        self.clf_sk = LinearDiscriminantAnalysis(n_components=p)
        self.clf.train(x, y, p)
        self.clf_sk.fit(x, y)
        np.testing.assert_equal(self.clf.cls, self.clf_sk.classes_)
        np.testing.assert_almost_equal(np.exp(self.clf.log_prob), self.clf_sk.priors_)
        #np.testing.assert_almost_equal(self.clf.transform_to_match_sk(x), self.clf_sk.transform(x))
        np.testing.assert_almost_equal(self.clf.predict(x), self.clf_sk.predict(x)) #
        np.testing.assert_almost_equal(self.clf.predict(x_new), self.clf_sk.predict(x_new))

    def testAgainstSkDocs(self):
        x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        x_new = np.array([[-0.8, -1]])
        self.assertAgainstSK(x, y, x_new)

    def testLowRank(self):
        # when m and n are close: low rank covariance matrix
        np.random.seed(0)
        x = np.random.randn(6, 4)
        y = np.random.randint(0, 3, size=(6,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)

    def testAgainstSkRandom(self):
        #np.random.seed(6)
        x = np.random.randn(6, 2)
        y = np.random.randint(0, 2, size=(6,))
        x_new = np.random.randn(5, 2)
        self.assertAgainstSK(x, y, x_new)
        #np.random.seed(5)
        x = np.random.randn(100, 4)
        y = np.random.randint(0, 3, size=(100,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)

    def testAgainstSkRandomDimRedTrans(self):
        x = np.random.randn(100, 8)
        y = np.random.randint(0, 6, size=(100,))
        x_new = np.random.randn(50, 8)
        clf_sk = LinearDiscriminantAnalysis(n_components=4)
        clf_sk.fit(x, y)
        clf = FDA()
        clf.train(x, y, p=4)
        x_trans = clf.transform_to_match_sk(x)
        cov = np.dot(x_trans.T, x_trans)
        x_trans_sk = clf_sk.transform(x)
        np.testing.assert_almost_equal(cov, np.dot(x_trans_sk.T, x_trans_sk))
        cov_diag = (1 - np.isclose(cov, 0))
        #testing that the cov matrix is diagonal
        np.testing.assert_almost_equal(cov_diag, np.identity(4))
        #np.testing(self.clf.predict(x), self.clf_sk.predict(x)) # Will not pass because sklearn lda does not predict on reduced dimension

class TestQDA(unittest.TestCase):
    def setUp(self):
        self.clf = QDA()
        self.clf_sk = QuadraticDiscriminantAnalysis()

    def assertAgainstSK(self, x, y, x_new):
        m, n = x.shape
        self.clf.train(x, y)
        self.clf_sk.fit(x, y)
        np.testing.assert_equal(self.clf.cls, self.clf_sk.classes_)
        # to test covariance: change m - nc to m in the code
        #np.testing.assert_almost_equal(np.linalg.inv(self.clf.S_w_inv), self.clf_sk.covariance_)
        #np.testing.assert_almost_equal(np.exp(self.clf.log_prob), self.clf_sk.priors_)
        np.testing.assert_almost_equal(self.clf.predict(x), self.clf_sk.predict(x)) #
        np.testing.assert_almost_equal(self.clf.predict(x_new), self.clf_sk.predict(x_new))

    def testAgainstSkDocs(self):
        x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
        y = np.array([1, 1, 1, 2, 2, 2])
        x_new = np.array([[-0.8, -1]])
        self.assertAgainstSK(x, y, x_new)

    def testOneSampleClass(self):
        # when m and n are close: mismatch covariances due to low rank (won't pass)
        np.random.seed(0)
        x = np.random.randn(6, 4)
        y = np.random.randint(0, 3, size=(6,))
        self.assertRaises(ZeroDivisionError, self.clf.train, x, y)

    def testSingularCov(self):
        np.random.seed(0)
        x = np.random.randn(6, 2)
        y = np.random.randint(0, 2, size=(6,))
        self.assertRaises(ZeroDivisionError, self.clf.train, x, y)

    def testAgainstSkRandom(self):
        #np.random.seed(5)
        x = np.random.randn(100, 4)
        y = np.random.randint(0, 3, size=(100,))
        x_new = np.random.randn(5, 4)
        self.assertAgainstSK(x, y, x_new)


if __name__ == '__main__':
    unittest.main()
