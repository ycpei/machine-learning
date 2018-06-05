#monte carlo k-arm bandit greedy strategy feasibility
#the probability that greedy strategy after one run of each arm of the k-arm bandit gives the optimal policy.
#question: is there a closed-form formula?
import numpy as np
import numpy.random as random

k = 10
N = 100000
mu = random.randn(N, k)
X = random.normal(mu, 1)

maxMuIdx = np.argmax(mu, axis=1)
maxXIdx = np.argmax(X, axis=1)

print(sum(maxMuIdx == maxXIdx) / N)
