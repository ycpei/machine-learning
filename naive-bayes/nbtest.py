from naive_bayes import *
import unittest
import numpy as np
from sklearn.naive_bayes import MultinomialNB

class TestNBMN(unittest.TestCase):
    def assertTupleAE(self, t1, t2):
        self.assertEqual(len(t1), len(t2))
        for x, y in zip(t1, t2):
            np.testing.assert_almost_equal(x, y)

    def assertTrainGeneric(self, x, y, output_expected):
        clf = NBMultinoulli()
        output_actual = clf.train(x, y)
        self.assertTupleAE(output_actual, output_expected)

    def assertPredictGeneric(self, x, y, x_new, output_expected):
        clf = NBMultinoulli()
        clf.train(x, y)
        output_actual = clf.predict(x_new)
        self.assertTupleAE(output_actual, output_expected)

    def testTrainSimple(self):
        x = [[1, 2, 1, 3], [2, 2, 2, 2]]
        y = np.array([0, 1])
        output_expected = (np.array([[0.5 , 0.25, 0.25], [0.  , 1.  , 0.  ]]), np.array([[0.5], [0.5]]))
        self.assertTrainGeneric(x, y, output_expected)
        x = [[1, 2, 1, 3], [1, 2, 2, 2]]
        y = np.array([0, 1])
        output_expected = (np.array([[0.5 , 0.25, 0.25], [0.25  , .75  , 0.  ]]), np.array([[0.5], [0.5]]))
        self.assertTrainGeneric(x, y, output_expected)

    def testPredictSimple(self):
        x = [[1, 2, 1, 3], [1, 2, 2, 2]]
        y = np.array([0, 1])
        x_new = [[1, 2], [2, 2], [3]]
        output_expected = [[0.4, 0.1, 1. ], [0.6, 0.9, 0. ]]
        self.assertPredictGeneric(x, y, x_new, output_expected)

    def testAgainstSk(self):
        np.random.seed(1)
        x = np.random.randint(5, size=(6, 20))
        y = np.arange(6)
        clf_sk = MultinomialNB(alpha=0)
        clf_sk.fit(x, y)
        xy_prob_expected = np.exp(clf_sk.feature_log_prob_)
        clf = NBMultinoulli()
        xy_prob_actual, _ = clf.train(x, y)
        np.testing.assert_almost_equal(xy_prob_actual, xy_prob_expected) #TODO: need to transform the output from sklearn otherwise this won't pass

if __name__ == '__main__':
    unittest.main()
