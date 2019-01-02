from shapley import shapley
import unittest
import numpy as np

class TestShapley(unittest.TestCase):
    def testOR1(self):
        """test feature contribution of x_1 | x_2 at x_1 = x_2 = 1
        """
        v = np.array([.75, 1, 1, 1])
        phi = shapley(2, v)
        phi_expected = np.array([.125, .125])
        np.testing.assert_almost_equal(phi, phi_expected)

    def testOR2(self):
        """test feature contribution of x_1 | x_2 at x_1 = 1, x_2 = 0
        """
        v = np.array([.75, 1, .5, 1])
        phi = shapley(2, v)
        phi_expected = np.array([.375, -.125])
        np.testing.assert_almost_equal(phi, phi_expected)

    def testAllOnes(self):
        """test the case where v(S) = s + 1: all phi's should be 1
        """
        n = 10
        v = np.zeros(2 ** n)
        for i in range(2 ** n):
            popcount = bin(i).count('1')
            v[i] = popcount + 1
        phi = shapley(n, v)
        phi_expected = np.ones(n)
        np.testing.assert_almost_equal(phi, phi_expected)

if __name__ == '__main__':
    unittest.main()
