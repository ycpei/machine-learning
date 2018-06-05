#monte carlo k-arm bandit greedy strategy feasibility
import numpy as np
import numpy.random as random

k = 10
N = 100000
mu = random.randn(N, k)
X = random.normal(mu, 1)

maxMuIdx = np.argmax(mu, axis=1)
maxXIdx = np.argmax(X, axis=1)

print(sum(maxMuIdx == maxXIdx) / N)
