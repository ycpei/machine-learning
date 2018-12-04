import numpy as np
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from da_sklearn import LinearDiscriminantAnalysis
from da import *

x = np.random.randn(100, 8)
y = np.concatenate([np.repeat(i, 20) for i in range(5)])
#y = np.random.randint(0, 6, size=(100,))
clf_sk = LinearDiscriminantAnalysis(n_components=4)
clf_sk.fit(x, y)
clf = FDA()
clf.train(x, y, p=4)
#print(clf.transform(x[:10,:]))
#print(clf_sk.transform(x[:10,:]))
#print(clf_sk.scalings / clf.trans)
#clf_sk = LinearDiscriminantAnalysis(n_components=2)
#clf_sk.fit(x, y)
#clf_sk1 = LinearDiscriminantAnalysis()
#clf_sk1.fit(x, y)
#print(all(clf_sk.predict(x) == clf_sk1.predict(x)))
