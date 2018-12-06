import numpy as np
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from da_sklearn import LinearDiscriminantAnalysis
from da import *

np.random.seed(0)
x = np.random.randn(6, 4)
y = np.random.randint(0, 3, size=(6,))
clf_sk = LinearDiscriminantAnalysis()
clf_sk.fit(x, y)

def old():
    x = np.random.randn(100, 8)
    x = x.astype(np.double)
    y = np.concatenate([np.repeat(i, 20) for i in range(5)])
    #y = np.random.randint(0, 6, size=(100,))
    clf_sk = LinearDiscriminantAnalysis(n_components=4)
    clf_sk.fit(x, y)
    clf = FDA()
    clf.train(x, y, p=4)
    np.testing.assert_almost_equal(clf.transform_to_match_sk(x[:10,:]), clf_sk.transform(x[:10,:]))

    #print(clf_sk.scalings / clf.trans)
    #clf_sk = LinearDiscriminantAnalysis(n_components=2)
    #clf_sk.fit(x, y)
    #clf_sk1 = LinearDiscriminantAnalysis()
    #clf_sk1.fit(x, y)
    #print(all(clf_sk.predict(x) == clf_sk1.predict(x)))
