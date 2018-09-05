'''
Author: Yuchen Pei (me@ypei.me)
Year: 2017
neural network
using conjugate gradient for optimisation
blackboxes:
    - scipy.optimize.minimize(method="cg")
data:
    - Kaggle MNIST data

note: single hidden layer, with 25 units, one-vs-all

Acknowledgements. thanks to:
    - Andrew Ng's ML course: https://www.coursera.org/learn/machine-learning
    - Hideki Ikeda's code: https://www.kaggle.com/hideki1234/neural-network/code
'''

import matplotlib.pyplot as plt
from numpy import *
from scipy.optimize import minimize
from pandas import read_csv, DataFrame
import unittest

class TestNN(unittest.TestCase):
    def test0(self):
        x, y = mnist(ifname="train1.csv")
        cx, cy = 0.98639455782312924, 0.80952380952380953
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    def test1(self):
        print("This may take up to 20 minutes")
        x, y = mnist(ifname="train.csv")
        cx, cy = 0.96669954760365995, 0.95032140306324897
        self.assertTrue(all(isclose((x, y), (cx, cy))))

    def test1(self):
        print("This may take up to 40 minutes")
        x, y = mnist(ifname="train.csv", maxiter=200)
        cx, cy = 0.98091771828973773, 0.95309896039996822
        self.assertTrue(all(isclose((x, y), (cx, cy))))

# data retrieval

def scaleX(x):
    return multiply(x, 1/255)

def addOnes(x):
    m, n = shape(x)
    return hstack((ones((m, 1)), x))

def num2IndMat(l):
    t = array(l)
    tt = [vectorize(int)((t == i)) for i in range(10)]
    return array(tt).T

def cutData(x):
    """
    70% training
    30% testing
    """
    m = len(x)
    t = int(m * .7)
    return x[:t], x[t:]


# sigmoid

def sigmoid(x):
    return 1 / (1 + e ** (-x))

def sigDot(a, theta):
    return sigmoid(dot(addOnes(a), theta))

def hTheta(theta1, theta2, x):
    """
    the result fo feedforward
    """
    a2 = sigDot(x, theta1)
    a3 = sigDot(a2, theta2)
    return addOnes(a2), a3

def costFunFromA(theta1, theta2, a3, y, lambda_):
    m = len(y)
    return - sum(1 / m * (y * log(a3) + (1 - y) * log (1 - a3)))\
           + sum(lambda_ / 2 / m * (theta1[1:] * theta1[1:])) + sum(lambda_ / 2 / m * (theta2[1:] * theta2[1:]))

def gradFromA(theta1, theta2, a1, a2, a3, y, lambda_):
    n1 = len(theta1) - 1
    n2 = len(theta1[0])
    n3 = len(theta2[0])
    m = len(y)
    dd1 = zeros((n1 + 1, n2))
    dd2 = zeros((n2 + 1, n3))
    for i in range(m):
        d3 = (a3[i] - y[i]).reshape((1, n3))
        d2 = (dot(d3, theta2.T) * a2[i] * (1 - a2[i])).reshape((1, n2 + 1))
        dd2 += a2[i].reshape((n2 + 1, 1)) * d3
        dd1 += (a1[i].reshape((n1 + 1, 1)) * d2)[:,1:]
    dd1[1:] += lambda_ * theta1[1:]
    dd2[1:] += lambda_ * theta2[1:]
    dd1 /= m
    dd2 /= m
    return dd1, dd2

def costFunAndGrad(theta1, theta2, x, y, lambda_):
    a2, a3 = hTheta(theta1, theta2, x)
    a1 = addOnes(x)
    return costFunFromA(theta1, theta2, a3, y, lambda_), gradFromA(theta1, theta2, a1, a2, a3, y, lambda_)

def costFunAndGradTheta(theta, x, y, lambda_, n1, n2, n3):
    theta1, theta2 = unpack(theta, n1, n2, n3)
    c, (dd1, dd2) = costFunAndGrad(theta1, theta2, x, y, lambda_)
    return c, pack(dd1, dd2)

def pack(x1, x2):
    return hstack((x1.reshape(x1.size), x2.reshape(x2.size)))

def unpack(x, n1, n2, n3):
    return x[:(n1 + 1) * n2].reshape((n1 + 1, n2)), x[(n1 + 1) * n2:].reshape((n2 + 1, n3))

def train(theta1, theta2, x, y, lambda_, maxiter):
    n1 = len(theta1) - 1
    n2 = len(theta1[0])
    n3 = len(theta2[0])
    theta = pack(theta1, theta2)
    return minimize(costFunAndGradTheta, theta, args=(x, y, lambda_, n1, n2, n3),
                   jac=True, method="CG", options={"maxiter": maxiter, "disp": True})

def predict(theta1, theta2, x):
    _, a3 = hTheta(theta1, theta2, x)
    return argmax(a3, axis=1)

def accuracy(theta1, theta2, x, l):
    t = (predict(theta1, theta2, x) == l)
    return sum(t) / size(t)
    
def randomizeTheta(m, n):
    epsilon = sqrt(6) / (sqrt(m) + sqrt(n))
    #print(epsilon)
    return 2 * epsilon * random.rand(m, n) - epsilon

def readSubData(ifname):
    iiter = getIiter(ifname)
    return scaleX(array([[int(y) for y in row] for row in iiter]))

def writeRows(ofname, l):
    f = open(ofname, 'w')
    f.write('ImageId,Label\n')
    for i, ll in enumerate(l):
        f.write(str(i + 1))
        f.write(',')
        f.write(str(ll))
        f.write('\n')
    f.close

def saveTheta(ofname, theta1, theta2):
    pf = open(ofname, 'wb')
    pickle.dump((theta1, theta2), pf)
    pf.close()

def mnist(ifname="train.csv", maxiter=100):
    # read data:
    df = read_csv("../data/kaggle-mnist/" + ifname)
    x = df.drop("label",axis=1).values
    l = df.label.values
    print("Finished reading data")

    #plotNums(5, x[:25], l[:25])
    #plt.show()

    # preprocess data:
    y = num2IndMat(l)
    trainX, testX = cutData(x)
    trainY, testY = cutData(y)
    trainL, testL = cutData(l)
    trainX = scaleX(trainX)
    testX = scaleX(testX)

    #initialise parameters:
    m, n1 = shape(trainX)
    n2 = 25
    n3 = 10
    #initTheta1 = zeros((n1 + 1, n2))
    #initTheta2 = zeros((n2 + 1, n3))
    initTheta1 = randomizeTheta(n1 + 1, n2)
    initTheta2 = randomizeTheta(n2 + 1, n3)
    lambda_ = 3

    # train:
    res = train(initTheta1, initTheta2, trainX, trainY, lambda_, maxiter)
    theta1, theta2 = unpack(res.x, n1, n2, n3)

    accuTrain = accuracy(theta1, theta2, trainX, trainL)
    accuTest = accuracy(theta1, theta2, testX, testL)

    # submit:
    df = read_csv("../data/kaggle-mnist/test.csv")
    subX = scaleX(df.values)
    subL = predict(theta1, theta2, subX)
    submission = DataFrame({"ImageId": list(range(1, len(subL) + 1)), "Label": subL})
    submission.to_csv("../data/kaggle-mnist/submit.csv", index=False, header=True)

    return accuTrain, accuTest


print(mnist(maxiter=200))
