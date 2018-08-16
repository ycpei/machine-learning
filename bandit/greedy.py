#monte carlo k-arm bandit greedy strategy feasibility
#the probability that greedy strategy after one run of each arm of the k-arm bandit gives the optimal policy.
#question: is there a closed-form formula?
import numpy as np
import numpy.random as random
import pandas as pd
from matplotlib import pyplot as plt

def greedy_prob(k, N=100000):
    mu = random.randn(N, k)
    X = random.normal(mu, 1)

    maxMuIdx = np.argmax(mu, axis=1)
    maxXIdx = np.argmax(X, axis=1)

    return sum(maxMuIdx == maxXIdx) / N

def generate_probs(maxK, minK=1):
    ks = range(minK, maxK)
    ps = [greedy_prob(k) for k in ks]
    df = pd.DataFrame({'k': ks, 'p': ps})
    df.to_csv('greedy-{}-{}.csv'.format(minK, maxK), index=False)

def plot_probs(ifname):
    df = pd.read_csv(ifname)
    ks, ps = df.k, df.p
    #ys = np.log(ks) / ks + .2
    #print(ps - 1 / ks)
    zs = 1 / ps ** 4
    #plt.plot(ks, ps)
    #c = (10000 - 256) / 81
    #b = 256 / 81 - 2 * c
    plt.plot(ks, zs)
    #plt.plot(ks, b * ks + c)
    plt.show()

def main():
    #generate_probs(100, 50)
    plot_probs('greedy-1-100.csv')

if __name__ == '__main__':
    main()
