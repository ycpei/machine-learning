from numpy import *
from scipy.linalg import norm, svd
from scipy.sparse.linalg import svds
from pandas import read_csv, DataFrame
import time

def example0():
    x = array([[3.0, 2.0, 2.0], [2.0, 3.0, -2.0]])
    s2 = sqrt(2)
    s18 = sqrt(18)
    cu = array([[1 / s2, 1 / s2], [1 / s2, - 1 / s2]])
    cs = array([5, 3])
    cvh = array([[1 / s2, 1 / s2, 0], [1 / s18, - 1 / s18, 4 / s18], [2 / 3, - 2 / 3, - 1 / 3]])
    csvd1 = hstack((cu.flatten(), cs.flatten(), cvh.flatten()))
    print(csvd1)
    cu1 = cu[:,0]
    cvh1 = cvh[0,:]
    cvh2 = cvh[0:2,:]
    cs1 = cs[0]
    u, s, vh = svd(x)
    print(svds(x.T, 1, return_singular_vectors="vh"))
    print(svds(x, 1, return_singular_vectors="u"))
    return compare(u, cu)
    #b1 = isclose((cu, cs, cvh), (u, s, vh)) or isclose((-cu, -cs, -cvh), (u, s, vh))
    #uu, ss, vvh = svd(x, full_matrices=False)
    #b2 = isclose((cu, cs, cvh2), (uu, ss, vvh)) or isclose((-cu, -cs, -cvh2), (uu, ss, vvh))
    #uuu, sss, vvvh = svds(x, 1)
    #b3 = isclose((cu1, cs1, cvh1), (uuu, sss, vvvh)) or isclose((-cu1, -cs1, -cvh1), (uuu, sss, vvvh))
    #return b1 and b2 and b3

def compare(x, y):
    return all(isclose(x, y)) or all(isclose(x, -y))

def pca(x, p):
    '''
    pca(x, p) = y, r
    fit x to p dimensions
    assuming x is normalised
    y is the resulting vectors
    r is the (1 - loss ratio of variance)
    '''
    m = len(x)
    var = norm(x) ** 2 / m
    _, s, vh = svd(x, full_matrices=False)
    v = vh[:p].T
    x1 = dot(x, v)
    var1 = norm(x1) ** 2 / m
    print(var1 / var)
    return x1, var1 / var

def pca1(x, p):
    '''
    pca(x, p) = y, r
    fit x to p dimensions
    assuming x is normalised
    y is the resulting vectors
    r is the (1 - loss ratio of variance)
    '''
    m = len(x)
    #starttime = time.time()
    var = norm(x) ** 2 / m
    #print("%s seconds computing the norm" % (time.time() - starttime))
    #starttime = time.time()
    _, _, vh = svds(x, p, return_singular_vectors="vh")
    #print("%s components: %s seconds with svds" % (p, (time.time() - starttime)))
    #starttime = time.time()
    v = array(vh[:p]).T
    x1 = dot(x, v)
    var1 = norm(x1) ** 2 / m
    #print("%s of variance retained: %s seconds computation" % (var1 / var, (time.time() - starttime)))
    return x1, var1 / var

def mnist():
    df = read_csv("../data/kaggle-mnist/train.csv")
    l = df.label.values
    x = df.drop("label", axis=1).values
    xm = multiply(x, 1 / 255)

    print("Finished reading data.")
    starttime = time.time()
    p = 50
    starttime = time.time()
    x1, t = pca1(xm, p)
    #print(shape(l), shape(x1))
    #dfMatrix = hstack((l.reshape(len(l), 1), x1))
    #dfLabels = ["label"] + ["comp{}".format(i) for i in range(p)]
    labelDict = {"label": l}
    compDict = dict(zip(["comp{}".format(i) for i in range(p)], x1.T))
    odf = DataFrame(dict(labelDict, **compDict))
    #odf = DataFrame(dict(zip(dflabels, dfmatrix.T)))
    ofname = "../data/kaggle-mnist/train-pca-{}-comp.csv".format(p)
    odf.to_csv(ofname, index=False, header=True)
    print("Written {} components to {}. Variance retention: {}. Time of computation: {} seconds".format(p, ofname, t, time.time() - starttime))
    

def mnist_example():
    df = read_csv("../data/kaggle-mnist/train.csv")
    l = df.label.values
    x = df.drop("label", axis=1).values
    xm = multiply(x, 1 / 255)

    #m = len(x)
    #meanx = sum(x, axis=0) / m
    #varx = norm(x) ** 2 / m
    #nx = (x - meanx) / sqrt(varx)

    print("Finished reading data.")
    for p in range(50, 700, 50):
        #starttime = time.time()
        pca1(xm, p)
        #print("%s seconds" % (time.time() - starttime))
    #pca1(xm, 100)
    #pca1(nx, 35)
    #pca(nx, 300)

print(mnist())
#print(example0())
