import numpy as np

def shapley(n, v):
    """compute the Shapley values
    inputs:
        n: int, number of players
        v: arr[float], 2 ** n: coalitional game worth
    outputs:
        arr[float], n: Shapley values
    """
    coef = np.zeros(n + 1)
    shapley = np.zeros(n)
    coef[0] = 1 / n
    for i in range(1, n - 1):
        coef[i] = i * coef[i - 1] / (n - i)
    coef[n - 1] = 1 / n
    for i in range(n):
        to_or = 1 << i
        for j in range(2 ** n):
            popcount = bin(j).count('1')
            shapley[i] += coef[popcount] * (v[j | to_or] - v[j])
    return shapley
