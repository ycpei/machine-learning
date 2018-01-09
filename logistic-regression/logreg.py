'''
Author: Yuchen Pei (me@ypei.me)
Year: 2017
linear logistic regression one-vs-all
using simple gradient descent or conjugate gradient
blackboxes:
    - scipy.optimize.fmin_cg
data: 
    - Kaggle MNIST data

note:
    - in gradient descent I optimised all the 10 units at once, without regularisation
    - in conjugate gradient I optimise each unit separately and combine the results for one-vs-all. It is faster this way. I also used regularisation

Acknowledgements. Thanks to:
    - Coursera ML course: https://www.coursera.org/learn/machine-learning
    - zgo2016's code on kaggle: https://www.kaggle.com/zgo2016/digit-recognizer-one-vs-all
'''


import matplotlib.pyplot as plt
from numpy import *
from scipy.optimize import fmin_cg
from pandas import read_csv, DataFrame
import unittest

### Tests

class TestLR(unittest.TestCase):
    def test0(self):
        x, y = mnist()
        cx, cy = 1.0, 0.82539682539682535
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    def test1(self):
        x, y = mnist(method="gd")
        cx, cy = 0.98979591836734693, 0.81746031746031744
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    #This test will take ~10mins
    def test2(self):
        x, y = mnist(ifname="train.csv")
        cx, cy = 0.93370522806898193, 0.91262598206491552
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    #This test will take ~10mins
    def test3(self):
        x, y = mnist(ifname="train.csv", method="gd")
        cx, cy = 0.8865607673730399, 0.88286643917149432
        self.assertTrue(all(isclose((x, y), (cx, cy))))

# Plotting

def plotNum(lab, img):
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title(lab)

def ceilDiv(x, y):
    return (x - 1) // y + 1

def plotNums(rowlen, im, l):
    collen = ceilDiv(len(im), rowlen)
    for i, (lab, img) in enumerate(zip(l, im)):
        plt.subplot(collen, rowlen, i + 1)
        plotNum(lab, img.reshape((28, 28)))
    #plt.tight_layout()

# Data preparation

def cutData(x, y, l):
    """
    70% training
    30% testing
    """
    m = len(x)
    t = int(m * .7)
    return x[:t], y[:t], l[:t], x[t:], y[t:], l[t:]

def num2IndMat(l):
    t = array(l)
    tt = [vectorize(int)((t == i)) for i in range(10)]
    return array(tt).T

def scaleX(x):
    m, n = shape(x)
    newX = ones((m, n + 1))
    newX[:,1:] = multiply(x, 1/255)
    return newX

# MNIST

def mnist(ifname="train1.csv", method="fmin_cg", maxiter=100):
    # read data:
    df = read_csv("../data/kaggle-mnist/" + ifname)
    x = df.drop("label",axis=1).values
    l = df.label.values
    print("Finished reading data")

    # plot data:
    #plotNums(5, x[:25], l[:25])
    #plt.show()

    # preprocess data:
    y = num2IndMat(l)
    trainX, trainY, trainL, testX, testY, testL = cutData(x, y, l)

    trainX = scaleX(trainX)
    testX = scaleX(testX)

    # initialise parameters:
    m = len(trainX)
    n = len(trainX[0]) - 1
    N = len(trainY[0])

    initTheta = zeros(((n + 1), N))

    # train:
    if method=="fmin_cg":
        theta = train2(initTheta, trainX, trainY, lambda_=.1) #100 iterations
    elif method=="gd":
        alpha = 5
        #epsilon = .01
        theta = train(initTheta, trainX, trainY, trainL, alpha)
    else:
        error("method should be either cg or gd.")

    # predict:
    accuTrain = accuracy(theta, trainX, trainL)
    accuTest = accuracy(theta, testX, testL)

    # submit:
    df = read_csv("../data/kaggle-mnist/test.csv")
    subX = scaleX(df.values)
    subL = [predict(theta, xx) for xx in subX]
    submission = DataFrame({"ImageId": list(range(1, len(subL) + 1)), "Label": subL})
    submission.to_csv("../data/kaggle-mnist/submit.csv", index=False, header=True)

    return accuTrain, accuTest

# Training

def sigmoid(x):
    return 1 / (1 + exp(-x))

def updateTheta(theta, x, y, alpha):
    return theta - alpha * gradient(theta, x, y)

def costFunction(theta, x, y):
    m, N = shape(y)
    sigxt = sigmoid(dot(x, theta))
    cm = (- y * log(sigxt) - (1 - y) * log(1 - sigxt)) / m / N
    return sum(cm)

def predict(theta, x):
    return argmax(sigmoid(dot(x, theta)))

def gradient(theta, x, y):
    m, N = shape(y)
    return dot(x.T, sigmoid(dot(x, theta)) - y) / m / N

def train(theta, x, y, l, alpha, maxiter=100):
    error = costFunction(theta, x, y)
    for i in range(maxiter):
        theta = updateTheta(theta, x, y, alpha)
        print("{}th iteration: accuracy {}".format(i, accuracy(theta, x, l)))
    print("final error:", costFunction(theta, x, y))
    return theta

def accuracy(theta, x, l):
    b = [1 if predict(theta, xx) == ll else 0 for xx, ll in zip(x, l)]
    return sum(b) / len(b)

def costFunction2(thetaCol, x, yCol, lambda_=0):
    sigxt = sigmoid(dot(x, thetaCol))
    m = len(x)
    return - 1 / m * sum(yCol * log(sigxt) + (1 - yCol) * log(1 - sigxt)) + 1 / 2 / m * lambda_ * sum(thetaCol[1:] * thetaCol[1:])

def gradient2(thetaCol, x, yCol, lambda_=0):
    m = len(x)
    return 1 / m * dot(x.T, (sigmoid(dot(x, thetaCol)) - yCol)) + 1 / m * lambda_ * insert(thetaCol[1:], 0, 0)

def train2(initTheta, x, y, lambda_=0, maxiter=100):
    newTheta = zeros(shape(initTheta))
    _, N = shape(y)
    for i in range(N):
        yCol = y[:,i]
        initThetaCol = initTheta[:,i]
        newThetaCol = fmin_cg(costFunction2, initThetaCol, fprime=gradient2, args=(x, yCol, lambda_), maxiter=maxiter)
        newTheta[:,i] = newThetaCol
    return newTheta


print(mnist(ifname="train.csv", method="fmin_cg"))
#unittest.main()
