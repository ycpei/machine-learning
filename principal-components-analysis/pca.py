from numpy import *
from scipy.linalg import norm, svd
from pandas import read_csv, DataFrame

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
    _, s, vh = svd(x)
    v = vh[:p].T
    x1 = dot(x, v)
    var1 = norm(x1) ** 2 / m
    print(var1 / var)
    return x1, var1 / var

df = read_csv("../data/kaggle-mnist/train.csv")
l = df.label.values
x = df.drop("label", axis=1).values
print("Finished reading data.")
for p in range(700, 300, -50):
    pca(x, p)
