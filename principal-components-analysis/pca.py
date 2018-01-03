"""
mnist data with PCA, tested with ../support-vector-machine/mnist-svm.py
blackbox: scipy.linalg.svds
acknowledgement:
    - Coursera ML course: https://www.coursera.org/learn/machine-learning
    - stanford cs229 lecture notes: http://cs229.stanford.edu/notes/cs229-notes3.pdf
    - MIT OpenCourseWare Linear Algebra course SVD lecture notes: https://ocw.mit.edu/courses/mathematics/18-06sc-linear-algebra-fall-2011/positive-definite-matrices-and-applications/singular-value-decomposition/MIT18_06SCF11_Ses3.5sum.pdf
"""

from numpy import *
from scipy.linalg import norm, svd
from scipy.sparse.linalg import svds
from pandas import read_csv, DataFrame
import time

#def example0():
#    x = array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
#    s2 = sqrt(2)
#    s18 = sqrt(18)
#    cu = array([[1 / s2, 1 / s2], [1 / s2, - 1 / s2]])
#    cs = array([5, 3])
#    cvh = array([[1 / s2, 1 / s2, 0], [1 / s18, - 1 / s18, 4 / s18], [2 / 3, - 2 / 3, - 1 / 3]])
#    csvd1 = hstack((cu.flatten(), cs.flatten(), cvh.flatten()))
#    print(csvd1)
#    cu1 = cu[:,0]
#    cvh1 = cvh[0,:]
#    cvh2 = cvh[0:2,:]
#    cs1 = cs[0]
#    u, s, vh = svd(x)
#    print(svds(x.T, 1, return_singular_vectors="vh"))
#    print(svds(x, 1, return_singular_vectors="u"))
#    return compare(u, cu)
#    #b1 = isclose((cu, cs, cvh), (u, s, vh)) or isclose((-cu, -cs, -cvh), (u, s, vh))
#    #uu, ss, vvh = svd(x, full_matrices=False)
#    #b2 = isclose((cu, cs, cvh2), (uu, ss, vvh)) or isclose((-cu, -cs, -cvh2), (uu, ss, vvh))
#    #uuu, sss, vvvh = svds(x, 1)
#    #b3 = isclose((cu1, cs1, cvh1), (uuu, sss, vvvh)) or isclose((-cu1, -cs1, -cvh1), (uuu, sss, vvvh))
#    #return b1 and b2 and b3

def compare(x, y):
    return all(isclose(x, y)) or all(isclose(x, -y))

def pcaBasis(x, p):
    _, _, vh = svds(x, p, return_singular_vectors="vh")
    return vh.T

def readData(ifname, isTrain):
    df = read_csv(ifname)
    if isTrain:
        l = df.label.values
        df1 = df.drop("label", axis=1)
    else:
        l = None
        df1 = df
    x = multiply(df1.values, 1 / 255)
    return x, l

def mnistGetData():
    x, l = readData("../data/kaggle-mnist/train.csv", isTrain=True)
    x1, _ = readData("../data/kaggle-mnist/test.csv", isTrain=False)
    return x, l, x1

def writeData(x, l, ofname):
    p = shape(x)[1]
    if l is None:
        labelDict = {}
    else:
        labelDict = {"label": l}
    compDict = dict(zip(["comp{}".format(i) for i in range(p)], x.T))
    odf = DataFrame(dict(labelDict, **compDict))
    odf.to_csv(ofname, index=False, header=True)
    print("Written {} components to {}".format(p, ofname))

def mnistWriteData(x, l, x1):
    p = shape(x)[1]
    writeData(x, l, "../data/kaggle-mnist/train-pca-{}-comp.csv".format(p))
    writeData(x1, None, "../data/kaggle-mnist/test-pca-{}-comp.csv".format(p))

def mnistPCA(p, x, x1):
    v = pcaBasis(x, p) 
    return dot(x, v), dot(x1, v)

def var(x):
    m = len(x)
    return norm(x) ** 2 / m

def varComp(x, xx):
    return var(xx) / var(x)

def mnist():
    x, l, x1 = mnistGetData()
    print("Finished reading data.")
    p = 35
    starttime = time.time()
    xx, xx1 = mnistPCA(p, x, x1)
    t, t1 = varComp(x, xx), varComp(x1, xx1)
    print("Finished reducing training and testing data to {} components."
          "Time of computation: {} seconds. Variance retention for training"
          "and testing sets are {} and {} respectively".format(p, time.time() - starttime, t, t1))
    mnistWriteData(xx, l, xx1)

mnist()
#print(example0())
