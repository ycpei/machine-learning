"""
k-means example. Using Coursera Machine Learning course's data.
blackboxes: none
acknowledgement: Coursera Machine Learning course
"""
import numpy as np
from scipy import linalg, io
import unittest

class TestKmeans(unittest.TestCase):
    def test0(self):
        self.assertTrue(example())

def label(data, centroids):
    # given data and centroids, return the labels
    distances = linalg.norm(np.expand_dims(centroids, axis=0) - np.expand_dims(data, axis=1), axis=2);
    return np.argmin(distances, axis=1);

def step(data, centroids):
    # return new centroids
    k = len(centroids)
    labels = label(data, centroids)
    return np.array([np.mean(data[labels == i], axis=0) for i in range(k)])

#def initialiseCentroids(k, data):

def kmeans(data):
    #k-means. return the optimal centroids and the labels of the datasets
    #initialise k centroids
    centroids = np.array([[3, 3], [6, 2], [8, 5]])
    done = False
    #step
    while not done:
    #for i in range(10):
        oldCentroids = centroids
        centroids = step(data, centroids)
        done = np.allclose(centroids, oldCentroids)
    return centroids, label(data, centroids)

def getData():
    #return the data of ex7data2.mat
    xx = io.loadmat('../data/ml-class/ex7data2.mat')
    return xx['X']

def example():
    data = getData()
    centroids, _ = kmeans(data)
    centroidsC = np.array([[ 1.95399466, 5.02557006],
                           [ 3.04367119, 1.01541041],
                           [ 6.03366736, 3.00052511]])
    return np.allclose(centroids, centroidsC)

unittest.main()
