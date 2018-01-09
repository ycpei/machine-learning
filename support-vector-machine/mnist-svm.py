'''
Author: Yuchen Pei (me@ypei.me)
Year: 2017
mnist with libsvm (scikit-learn)
blackboxes:
    - sklearn.svm.SVC
data: mnist data from kaggle

Acknowledgements. Thanks to:
    - Mr Mean's kaggle code: https://www.kaggle.com/mrmean/svm-trial
'''

from numpy import *
from scipy import linalg, io
import matplotlib.pyplot as plt
import unittest
from pandas import read_csv, DataFrame
from sklearn.svm import SVC
import time

class TestSVM(unittest.TestCase):

    def test0(self):
        x, y = mnist(ifname="train1.csv")
        cx, cy = (1.0, 0.84126984126984128)
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    def test0(self):
        print("This may take up to 20 minutes.")
        x, y = mnist(ifname="train.csv")
        cx, cy = (0.99959182285111736, 0.97952543448932627)
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

def mnist(ifname, ifname1, scale=True):
    workDir = "../data/kaggle-mnist/"
    trainX, trainL, testX, testL = readData(workDir + ifname)

    startTime = time.time()
    if scale:
        trainX = scaleX(trainX)
        testX = scaleX(testX)

    clf = SVC(gamma=.01, C=10)
    clf.fit(trainX, trainL)

    predTrainL = clf.predict(trainX)
    predTestL = clf.predict(testX)

    accuTrain = accuracy(predTrainL, trainL)
    accuTest = accuracy(predTestL, testL)

    # submit:
    ofname = workDir + "submit.csv"
    df = read_csv(workDir + ifname1)
    if scale:
        subX = scaleX(df.values)
    else:
        subX = df.values
    subL = clf.predict(subX)
    submission = DataFrame({"ImageId": list(range(1, len(subL) + 1)), "Label": subL})
    submission.to_csv(ofname, index=False, header=True)

    print("Finished training and predicting. Total time: {}. Prediction written to {}."
          "Training set accuracy: {}. Testing set accuracy: {}".format(time.time() - startTime, 
              ofname, accuTrain, accuTest))


#print(mnist(ifname="train.csv", ifname1="test.csv", scale=True))
mnist(ifname="train-pca-35-comp.csv", ifname1="test-pca-35-comp.csv", scale=False)
