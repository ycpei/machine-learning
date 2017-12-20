import csv
from numpy import *
from scipy.optimize import minimize
import pickle

# data retrieval

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

def cutData(x):
    """
    70% training
    30% testing
    """
    m = len(x)
    t = int(m * .7)
    return x[:t], x[t:]

def num2IndMat(l):
    t = array(l)
    tt = [vectorize(int)((t == i)) for i in range(10)]
    return array(tt).T

def scaleX(x):
    return multiply(x, 1/255)

def addOnes(x):
    m, n = shape(x)
    return hstack((ones((m, 1)), x))

def readData(ifname):
    x, l = getAllRows(ifname)
    return scaleX(x), num2IndMat(l), l

def readData1(ifname):
    x, y, l = readData(ifname)
    trainX, testX = cutData(x)
    trainY, testY = cutData(y)
    trainL, testL = cutData(l)
    return trainX, trainY, trainL, testX, testY, testL

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
    return - sum(1 / m * (y * log(a3) + (1 - y) * log (1 - a3)))\
           + sum(lambda_ / 2 / m * (theta1[1:] * theta1[1:])) + sum(lambda_ / 2 / m * (theta2[1:] * theta2[1:]))

def gradFromA(theta1, theta2, a1, a2, a3, y, lambda_):
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

def costFunAndGradTheta(theta, x, y, lambda_):
    theta1, theta2 = unpack(theta)
    c, (dd1, dd2) = costFunAndGrad(theta1, theta2, x, y, lambda_)
    return c, pack(dd1, dd2)

def pack(x1, x2):
    return hstack((x1.reshape(x1.size), x2.reshape(x2.size)))

def unpack(x):
    return x[:(n1 + 1) * n2].reshape((n1 + 1, n2)), x[(n1 + 1) * n2:].reshape((n2 + 1, n3))

def train(theta1, theta2, x, y, lambda_):
    theta = pack(theta1, theta2)
    return minimize(costFunAndGradTheta, theta, args=(x, y, lambda_),
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

trainX, trainY, trainL, testX, testY, testL = readData1('../data/kaggle-mnist/rain.csv')
subX = readSubData('../data/kaggle-mnist/est.csv')

m, n1 = shape(trainX)
n2 = 25
n3 = 10
#initTheta1 = zeros((n1 + 1, n2))
#initTheta2 = zeros((n2 + 1, n3))
initTheta1 = randomizeTheta(n1 + 1, n2)
initTheta2 = randomizeTheta(n2 + 1, n3)
lambda_ = 3
maxiter = 100
#theta = pack(initTheta1, initTheta2)
#print(costFunAndGrad(initTheta1, initTheta2, trainX, trainY, lambda_))
res = train(initTheta1, initTheta2, trainX, trainY, lambda_)
theta1, theta2 = unpack(res.x)

print(accuracy(theta1, theta2, trainX, trainL))
print(accuracy(theta1, theta2, testX, testL))
subL = predict(theta1, theta2, subX)
writeRows('submission.csv', subL)

saveTheta('theta_nn_100_1.pkl', theta1, theta2)

res = train(theta1, theta2, trainX, trainY, lambda_)
theta1, theta2 = unpack(res.x)

print(accuracy(theta1, theta2, trainX, trainL))
print(accuracy(theta1, theta2, testX, testL))
subL = predict(theta1, theta2, subX)
writeRows('submission.csv', subL)
