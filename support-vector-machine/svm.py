'''
Author: Yuchen Pei (me@ypei.me)
Year: 2017
support vector machine via sequential minimal optimisation (smo)
blackboxes:
    - none
data:
    - simple 2d data
    - ex6data1.mat and ex6data2.mat from Coursera ML course
    - mnist data from kaggle
Acknowledgements. Thanks to:
    - Coursera ML course: https://www.coursera.org/learn/machine-learning
    - Platt's paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-98-14.pdf
    - Stanford CS229 lecturenotes:
        - http://cs229.stanford.edu/materials/smo.pdf
        - http://cs229.stanford.edu/notes/cs229-notes3.pdf

note: this file is for pedagogical purposes only. training the complete mnist data (with 42000 samples) is very slow. use libsvm instead.
'''

from numpy import *
from scipy import linalg, io
import matplotlib.pyplot as plt
import unittest
import csv
import sys

### Tests

class TestSVM(unittest.TestCase):

    def test0(self):
        self.assertTrue(example0())

    def test1(self):
        self.assertTrue(example1())

    def test2(self):
        self.assertTrue(example2())

    def test3(self):
        self.assertTrue(example3())

    def test4(self):
        self.assertTrue(example4())

    def test5(self):
        x, y = mnist()
        cx, cy = 1.0, 0.10317460317460317
        self.assertTrue(all(isclose((x, y), (cx, cy))))

def plotDots(x, y):
    xpos = []
    ypos = []
    xneg = []
    yneg = []
    for [xx, yy], zz in zip(x, y):
        if zz == 1:
            xpos.append(xx)
            ypos.append(yy)
        else:
            xneg.append(xx)
            yneg.append(yy)
    plt.scatter(xpos, ypos, color='g')
    plt.scatter(xneg, yneg, color='r')

def example0():
    x = array([[0, -1], [-1, 1], [1, 1]])
    y = array([-1, 1, 1])
    m = len(y)
    c = 1
    initB = 0
    initAlpha = zeros(m)
    alpha, b = go(x, y, c, initB, initAlpha, kernellinear, test=True)
    #alpha, b = go(x, y, c, initB, initAlpha)
    return linalg.norm(alpha - [.5, .25, .25]) + abs(b) < testTol

def example1():
    x = array([[0, 1], [0, -1]])
    y = array([1, -1])
    m = len(y)
    c = 1
    initB = 0
    initAlpha = zeros(m)
    alpha, b = go(x, y, c, initB, initAlpha, kernellinear, test=True)
    return linalg.norm(alpha - [.5, .5]) + abs(b) < testTol

def example2():
    x = array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    y = array([1, 1, -1, -1])
    m = len(y)
    c = 1
    initB = 0
    initAlpha = zeros(m)
    alpha, b = go(x, y, c, initB, initAlpha, kernellinear, test=True)
    return linalg.norm(alpha - [.5, .5, .5, .5]) + abs(b) < testTol

def example3():
    xx = io.loadmat('../data/ml-class/ex6data1.mat')
    x = xx['X']
    y = array([1 if yy == 1 else -1 for yy in xx['y']])
    m = len(y)
    c = 1
    initB = 0
    initAlpha = zeros(m)
    alpha, b = go(x, y, c, initB, initAlpha, kernellinear, test=True)
    #checkPred(x, y, alpha, b)
    aa = array([ 0.,  0.,  0.,  0.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  1.,  0.56789664,  0.,  1.,
        0.,  0.,  0.,  0.,  0.92987214,
        1.,  1.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.49776878,  0.,  0.,
        0.,  0.,  1.,  0.,  0.,  1.        ])
    bb = -10.3440152
    return linalg.norm(alpha - aa) + abs(b - bb) < testTol

def example4():
    print("This may take 1 minute.")
    xx = io.loadmat('../data/ml-class/ex6data2.mat')
    x = xx['X']
    y = array([1 if yy == 1 else -1 for yy in xx['y']])
    m = len(y)
    c = 1
    initB = 0
    initAlpha = zeros(m)
    alpha, b = go(x, y, c, initB, initAlpha, kernelrbf, test=True)
    aa = array([ 0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  1.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        1.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  0.,
        1.,  0.70892422,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,
        1.,  1.,  1.,  0.75321628,  1.,
        0.,  0.,  0.,  1.,  1.,
        0.,  0.,  0.,  0.,  1.,
        0.,  0.,  0.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  1.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.51334936,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  1.,  1.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        1.,  0.,  1.,  1.,  1.,
        0.84786339,  0.,  0.,  1.,  1.,
        1.,  1.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.76556368,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  1.,
        0.,  0.,  0.65363792,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.92247842,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,
        1.,  0.,  1.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.23847057,
        1.,  1.,  1.,  1.,  0.,
        0.,  0.,  1.,  1.,  1.,
        1.,  1.,  0.,  0.,  0.,
        0.,  0.,  1.,  1.,  0.,
        0.,  0.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.9151932,  0.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.30489063,
        1.,  0.780191,  0.,  1.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  0.,  0.,
        0.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.90118877,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  1.,
        0.,  1.,  1.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  1.,  1.,  1.,
        1.,  1.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  0.,
        0.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  1.,  1.,
        0.,  0.,  0.71278119,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,
        1.,  0.,  1.,  1.,  1.,
        1.,  0.40717627,  1.,  1.,  1.,
        1.,  1.,  0.,  0.,  0.,
        0.,  0.,  1.,  1.,  0.,
        0.,  0.,  0.,  1.,  1.,
        1.,  1.,  0.,  0.8897459,  0.,
        0.,  0.,  1.,  1.,  1.,
        1.,  1.,  0.02731483,  0.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.25828378,
        0.00865037,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.29936278,  1.,  1.,
        1.,  1.,  1.,  1.,  1.,
        1.,  0.,  0.,  0.,  0.,
        1.,  1.,  1.,  0.,  0.,
        0.,  0.,  1.,  1.,  1.,
        0.22152315,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.80664894,
        0.,  0.,  0.,  0.,  0.,
        0.,  1.,  0.,  0.,  0.,
        0.25060722,  1.,  1.,  1.,  0.,
        0.,  1.,  0.,  1.,  1.,
        1.,  1.,  1.,  1.,  0.,
        0.,  1.,  1.,  1.,  1.,
        1.,  0.,  0.,  0.,  1.,
        1.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  1.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  0.,  0.,
        0.,  0.,  0.,  1.,  0.,
        0.,  0.,  0.39401826,  1.,  0.0837919,
        0.,  0.,  1.,  0.,  0.,
        0.,  0.,  0.        ])
    bb = 0.40375059
    return linalg.norm(alpha - aa) + abs(b - bb) < testTol

def mnist():
    trainX, trainY, trainL, testX, testY, testL = readData1("../data/kaggle-mnist/train1.csv")
    kernel = kernelrbf
    m = len(trainX)
    print("Data retrieved. Training over", m, "images")
    initAlpha = zeros(m)
    initB = 0
    c = 1
    p = 10
    alpha = zeros((m, p))
    error = zeros((m, p))
    trainP = zeros((m, p))
    b = zeros(p)

    for i in range(p):
        y = trainY[:,i]
        initError = computeError(trainX, y, initAlpha, initB, kernel)
        print("Error computed for round", i)
        alpha[:,i], b[i], error[:,i] = smo(trainX, y, c, initAlpha, initB, initError, kernel)
        trainP[:,i] = error[:,i] + trainY[:,i] #One can compare the trainP obtained this way with the following line and see they are the same
        #trainP[:,i] = array([predict1(xx, trainX, trainY[:,i], alpha[:,i], b[i], kernel) for xx in trainX])
        print("SMO round", i, "complete.")
    trainAccu = accuracy(pred2Digits(trainP), trainL)

    testP = zeros(shape(testY))
    for i in range(p):
        testP[:,i] = array([predict1(xx, trainX, trainY[:,i], alpha[:,i], b[i], kernel) for xx in testX])
    testAccu = accuracy(pred2Digits(testP), testL)

    return trainAccu, testAccu


### read the data from csv

def getIiter(ifname):
    """
    Get the iterator from a csv file with filename ifname
    """
    ifile = open(ifname, 'r')
    iiter = csv.reader(ifile)
    iiter.__next__()
    return iiter

def getRow(iiter):
    """
    Get one line from a csv iterator
    """
    return parseRow(iiter.__next__())

def parseRow(s):
    y = [int(x) for x in s]
    lab = y[0]
    z = y[1:]
    return lab, z

def getRows(n, iiter):
    """
    Get the first n rows
    """
    x = []
    for i in range(n):
        x.append(getRow(iiter))
    return x

def getAllRows(ifname):
    iiter = getIiter(ifname)
    x = []
    l = []
    for row in iiter:
        lab, z = parseRow(row)
        x.append(z)
        l.append(lab)
    return x, l

def cutData(x, trainRatio=.7):
    """
    70% training
    30% testing
    """
    m = len(x)
    t = int(m * trainRatio)
    return x[:t], x[t:]

def num2IndMat(l):
    def f(x):
        if x:
            return 1
        else:
            return -1
    t = array(l)
    tt = [vectorize(f)((t == i)) for i in range(10)]
    return array(tt).T

def scaleX(x):
    return multiply(x, 1/255)

def readData(ifname):
    x, l = getAllRows(ifname)
    return scaleX(x), num2IndMat(l), l

def readData1(ifname):
    x, y, l = readData(ifname)
    trainX, testX = cutData(x)
    trainY, testY = cutData(y)
    trainL, testL = cutData(l)
    return trainX, trainY, trainL, testX, testY, testL

### Main function to run svm

def go(x, y, c, initB, initAlpha, kernel, test=True):

    if not test:
        plotDots(x, y)
        plt.show()

    initError = computeError(x, y, initAlpha, initB, kernel)
    alpha, b, error = smo(x, y, c, initAlpha, initB, initError, kernel)

    print("smo complete.")

    if not test:
        #plotLinear(x, y, alpha, b)
        plotBoundary(x, y, alpha, b, kernel)
        plt.show()

    return alpha, b

### Plotting

def plotLine(theta, x):
    xmin = min(x)
    xmax = max(x)
    xr = arange(xmin, xmax, (xmax - xmin) / 100)
    yr = [theta[0] + x * theta[1] for x in xr]
    plt.plot(xr, yr)

def plotLinear(x, y, alpha, b):
    plotDots(x, y)
    w = wFromALinear(alpha, x, y)
    plotLine([- b / w[1], - w[0] / w[1]], x[:,0])

def checkPred(x, y, alpha, b):
    w = wFromALinear(alpha, x, y)
    ok = True
    x1 = x[:,0]
    x2 = x[:,1]
    x1r = linspace(min(x1), max(x1), num=100)
    x2r = linspace(min(x2), max(x2), num=100)
    for x1a in x1r:
        for x2a in x2r:
            p1 = dot(w, array([x1a, x2a])) + b
            p2 = predict1([x1a, x2a], x, y, alpha, b)
            if not ((p1 < p2 + .00001) and (p1 > p2 - .00001)):
                ok = False
                print(x1a, x2a, p1, p2)
    if ok:
        print("predictions agree.")
    else:
        print("predictions disagree!")

def testContour():
    n = 10
    xr = linspace(-2, 2, num=n)
    yr = linspace(-2, 2, num=2 * n)
    z = array([[x + 2 * y for y in yr] for x in xr]).reshape(n, 2 * n)
    plt.contour(xr, yr, z, levels=[0])
    plt.show()

def plotBoundary(x, y, alpha, b, kernel):
    n = 100
    w = wFromALinear(alpha, x, y)
    plotDots(x, y)
    x1 = x[:,0]
    x2 = x[:,1]
    x1r = linspace(min(x1), max(x1), num=n)
    x2r = linspace(min(x2), max(x2), num=n)
    z = array([[predict1([xx1, xx2], x, y, alpha, b, kernel) for xx1 in x1r] for xx2 in x2r]).reshape((n, n))
    plt.contour(x1r, x2r, z, levels=[0])

def wFromALinear(alpha, x, y):
    return dot(x.T, alpha * y)

### Computation and algorithm implementation
def pred2Digits(pred):
    return argmax(pred, axis=1)

def accuracy(pred, y):
    t = (pred == y)
    return sum(t) / size(t)

def predict1(xx, x, y, alpha, b, kernel):
    return sum([ai * yi * kernel(xi, xx) for ai, xi, yi in zip(alpha, x, y)]) + b

def computeError(x, y, alpha, b, kernel):
    '''
    compute the error array
    '''
    m = len(y)
    k = array([[kernel(x[i], x[j]) for j in range(m)] for i in range(m)])
    ay = (alpha * y).reshape(m, 1)
    return dot(k, ay) + b - y.reshape(m, 1)

def kernelrbf(x, y, sigma=0.1):
    '''
    gaussian kernel with variance 1
    '''
    return e ** (- linalg.norm(x - y) ** 2 / (2 * sigma ** 2))

def kernellinear(x, y):
    '''
    linear kernel
    '''
    return dot(x, y)

def smo(x, y, c, alpha, b, error, kernel):
    '''
    smo(x, y, c, alpha, b, error) = alpha, b
    sum alpha[i] - 1 / 2 sum (y[i] y[j]) alpha[i] alpha[j] k[i][j]
    subject to 0 <= alpha[i] <= c and sum alpha[i] y[i] = 0
    '''

    # the inner loop selecting the second coordinate
    def innerLoop():
        nonlocal b, error
        s = sorted(range(len(error)), key=lambda j: - abs(error[j] - ei))
        for j in s:
            if j == i:
                print("something strange happened")
                print(i, s, error)
                sys.exit()
            a1, a2, nb = smoIte(x[i], x[j], y[i], y[j], alpha[i], alpha[j], error[i], error[j], b, c, kernel)
            if (a1, a2, nb) != (alpha[i], alpha[j], b):
                error = [(a1 - alpha[i]) * y[i] * kernel(x[i], x[k]) + 
                         (a2 - alpha[j]) * y[j] * kernel(x[j], x[k]) + nb - b +
                         error[k] for k in range(m)]
                alpha[i], alpha[j], b = a1, a2, nb
                break

    done = False
    m = len(y)
    while not done:
        done = True
        # first pass on all alphas
        for i in range(m):
            ai, yi, ei = alpha[i], y[i], error[i]
            if (ai < c and yi * ei < - tol) \
                    or (ai > 0 and yi * ei > tol):
                done = False
                innerLoop()

        # then passes on all nonbound alphas
        done1 = False
        while not done1:
            done1 = True
            for i in range(m):
                ai, yi, ei = alpha[i], y[i], error[i]
                if (ai < c and ai > 0 and abs(ei) > tol):
                    done1 = False
                    innerLoop()

    return alpha, b, error

def smoIte(x1, x2, y1, y2, a1, a2, e1, e2, b, c, kernel):
    '''
    smoIte(...) = a1, a2, e1, e2, b: one iteration
    '''
    #get all the values
    oldArgs = a1, a2, b

    #get the kernel values
    k11 = kernel(x1, x1)
    k12 = kernel(x1, x2)
    k22 = kernel(x2, x2)
    eta = k11 - 2 * k12 + k22
    if eta <= 0:
        return oldArgs

    #get the bounds
    if y1 == y2:
        l = max(0, a1 + a2 - c)
        h = min(a1 + a2, c)
    else:
        l = max(0, a2 - a1)
        h = c - max(0, a1 - a2)
    if l == h:
        return oldArgs

    #get new alpha1 alpha2
    a2a = a2 + (y2 / eta) * (e1 - e2)
    if a2a > h:
        a2b = h
    elif a2a < l:
        a2b = l
    else:
        a2b = a2a
    if a2b == a2:
        return oldArgs
    a1b = a1 + y1 * y2 * (a2 - a2b)

    #calculate the new threshold and new errors
    b1 = b - e1 - y1 * (a1b - a1) * k11 - y2 * (a2b - a2) * k12
    b2 = b - e2 - y1 * (a1b - a1) * k12 - y2 * (a2b - a2) * k22
    if tol < a1b < c - tol:
        bb = b1
    elif tol < a2b < c - tol:
        bb = b2
    else:
        bb = (b1 + b2) / 2

    return a1b, a2b, bb

testTol = .001
tol = .001
#print(example1())
unittest.main() #tol was set to be .001 in the tests
#print(mnist())
#testContour()
