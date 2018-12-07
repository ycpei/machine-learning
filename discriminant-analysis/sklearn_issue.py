#https://github.com/scikit-learn/scikit-learn/issues/12731
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from da import *

def k():
    x = np.random.randn(16, 4)
    _, s, vt = np.linalg.svd(x)
    x = np.dot(x, vt.T / s)
    print(np.dot(x.T, x))

def p():
    n_classes = 4
    n_samples = 16
    n_features = 6
    #np.random.seed(0)
    x = np.random.randn(n_samples, n_features)
    y = np.random.randint(n_classes, size=n_samples)
    #class_size = n_samples // n_classes
    #y = np.concatenate([[i for _ in range(class_size)] for i in range(n_classes)])
    #print(x_trans.dtype)
    clf = FDA()
    clf.train(x, y, p=3)
    #x_trans1 = np.dot(x - clf.mu[y], clf.trans.T) #OK
    x_trans2 = np.dot(x - clf.xbar, clf.trans_proj.T)
    x_trans3 = np.dot(x - clf.mu[y], clf.trans_proj.T)
    #print(np.dot(x_trans1.T, x_trans1) / (n_samples - n_classes))
    print(np.dot(x_trans2.T, x_trans2) / (n_samples - n_classes)) #OK
    print(np.dot(x_trans3.T, x_trans3) / (n_samples - n_classes)) #OK
    clf_sk = LinearDiscriminantAnalysis(n_components=3)
    clf_sk.fit(x, y)
    #x_trans4 = np.dot(x - clf.mu[y], clf_sk.scalings_[:, :3])
    #x_trans5 = np.dot(x - clf.xbar, clf_sk.scalings_[:, :3])
    x_trans6 = np.dot(clf.mu - clf.xbar, clf_sk.scalings_[:, :3])
    #print(np.dot(x_trans4.T, x_trans4) / (n_samples - n_classes)) #OK
    #print(np.dot(x_trans5.T, x_trans5) / (n_samples - n_classes)) #orthogonal
    print(np.dot(x_trans6.T, x_trans6) / (n_samples - n_classes)) #not orthogonal

def g():
    n_classes = 4
    n_samples = 16
    n_features = 6
    #np.random.seed(0)
    x = np.random.randn(n_samples, n_features)
    class_size = n_samples // n_classes
    #y = np.concatenate([[i for _ in range(class_size)] for i in range(n_classes)])
    y = np.concatenate([np.repeat(i, x) for i, x in enumerate([1, 3, 5, 7])])
    #y = np.random.randint(n_classes, size=n_samples)
    clf = LinearDiscriminantAnalysis()
    clf.fit(x, y)
    x_trans = clf.transform(x)
    print(np.dot(x_trans.T, x_trans) / (n_samples - n_classes)) 
    clf_red = LinearDiscriminantAnalysis(n_components=3)
    clf_red.fit(x, y)
    x_trans = clf_red.transform(x)
    #print(x_trans.dtype)
    print(np.dot(x_trans.T, x_trans) / (n_samples - n_classes))
    clf = FDA()
    clf.train(x, y, p=3)
    x_trans = clf.transform_to_match_sk(x)
    #print(x_trans.dtype)
    print(np.dot(x_trans.T, x_trans) / (n_samples - n_classes))
    x_trans1 = np.dot(clf.mu[y] - clf.xbar, clf.trans_proj.T)
    print(np.dot(x_trans1.T, x_trans1) / (n_samples - n_classes))

def f():
    n_classes = 2
    n_samples = 8
    n_features = 4
    np.random.seed(0)
    x = np.random.randn(n_samples, n_features)
    mid = n_samples // 2
    x[:mid] = x[:mid] - np.mean(x[:mid], axis=0)
    x[mid:] = x[mid:] - np.mean(x[mid:], axis=0)
    x1 = x / np.sqrt(n_samples - n_classes)
    std = np.std(x, axis=0)
    x2 = x1 / std
    _, s1, vt1 = np.linalg.svd(x1, full_matrices=0)
    _, s2, vt2 = np.linalg.svd(x2, full_matrices=0)
    vt2 = vt2 / std
    scaling1 = vt1.T / s1
    scaling2 = vt2.T / s2
    print("scaling1:", scaling1)
    print("scaling2:", scaling2)

def h():
    n_classes = 2
    n_samples = 10
    n_features = 4
    np.random.seed(0)
    x = np.random.randn(n_samples, n_features)
    mid = n_samples // 2
    xc = np.zeros((n_samples, n_features))
    xc[:mid] = x[:mid] - np.mean(x[:mid], axis=0)
    xc[mid:] = x[mid:] - np.mean(x[mid:], axis=0)
    #print(np.mean(x[:mid], axis=0))
    #print(np.mean(x[mid:], axis=0))
    x1 = xc / np.sqrt(n_samples - n_classes)
    std = np.std(xc, axis=0)
    x2 = x1 / std
    _, s1, vt1 = np.linalg.svd(x1, full_matrices=0)
    _, s2, vt2 = np.linalg.svd(x2, full_matrices=0)
    vt2 = vt2 / std
    scaling1 = vt1.T / s1
    scaling2 = vt2.T / s2
    print("scaling1:", scaling1)
    print("scaling2:", scaling2)
    xbar = np.mean(x, axis=0)
    x_trans1 = np.dot(xc, scaling1)
    x_trans2 = np.dot(xc, scaling2)
    print("dot1:", np.dot(x_trans1.T, x_trans1) / (n_samples - n_classes))
    print("dot2:", np.dot(x_trans2.T, x_trans2) / (n_samples - n_classes))

g()
