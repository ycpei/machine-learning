'''
mnist with libsvm (scikit-learn)
blackboxes:
    - sklearn.svm.SVC
'''

from numpy import *
from scipy import linalg, io
import matplotlib.pyplot as plt
import unittest
from pandas import read_csv, DataFrame
from sklearn.svm import SVC

class TestSVM(unittest.TestCase):

    def test0(self):
        x, y = mnist(ifname="train1.csv")
        cx, cy = 0.68367346938775508, 0.58730158730158732
        self.assertTrue(all(isclose((x, y), (cx, cy))))

def num2IndMat(l):
    t = array(l)
    tt = [vectorize(int)((t == i)) for i in range(10)]
    return array(tt).T

def scaleX(x):
    return multiply(x, 1/255)

def cutData(x):
    """
    70% training
    30% testing
    """
    m = len(x)
    t = int(m * .7)
    return x[:t], x[t:]

def readData(ifname):
    df = read_csv(ifname)
    x = df.drop("label", axis=1).values
    l = df.label.values
    print("Finished reading data.")
    trainX, testX = cutData(x)
    trainL, testL = cutData(l)
    return trainX, trainL, testX, testL

def accuracy(x, y):
    t = (x == y)
    return sum(t) / len(t)

def mnist(ifname):
    trainX, trainL, testX, testL = readData("../data/kaggle-mnist/" + ifname)

    trainX = scaleX(trainX)
    testX = scaleX(testX)

    clf = SVC()
    clf.fit(trainX, trainL)

    predTrainL = clf.predict(trainX)
    predTestL = clf.predict(testX)

    accuTrain = accuracy(predTrainL, trainL)
    accuTest = accuracy(predTestL, testL)

    return accuTrain, accuTest

print(mnist(ifname="train.csv"))
