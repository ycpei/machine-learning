import numpy as np

class NBGaussian:
    """Gaussian NB
    """
    def __init__(self):
        pass

    def train(self, x, y, var_smoothing=1e-9):
        """train a model
        inputs:
            x: array[[float]], m x n
            y: array[int], m
        outputs:
        modifies:
            self.mu, self.sigma, self.cls, self.icls, self.y_prob
        """
        self.cls = list(set(y))
        #self.icls = {c: i for i, c in enumerate(cls)}
        nc = len(self.cls)
        m, n = x.shape
        self.mu = np.zeros((nc, n))
        self.sigma2 = np.zeros((nc, n))
        self.y_prob = np.zeros(nc)
        for i, c in enumerate(self.cls):
            count = np.sum(y == c)
            self.y_prob[i] = count / m
            self.mu[i, :] = np.sum(x[y == c], axis=0) / count
            self.sigma2[i, :] = np.sum((x[y == c] - self.mu[i, :]) ** 2, axis=0) / count
        epsilon = var_smoothing * np.max(np.var(x, axis=0))
        self.sigma2 += epsilon

    def predict(self, x):
        """predict
        inputs:
            x: array[[float]], m x n
        outputs:
            y: array[int], m
        """
        nc = len(y_prob)
        m, n = x.shape
        log_probs = np.zeros((m, nc))
        for i in range(m):
            sample = x[i, :]
            log_probs[i, :] = np.sum(- np.log(self.sigma2) / 2 - (sample - self.mu) ** 2 / self.sigma2, axis=1)
        y = [self.cls[i] for i in np.argmax(log_probs, axis=1)]
        return y


class NBMultinoulli:
    """multinoulli NB with Laplace smoothing
    """
    def __init__(self, alpha=0):
        self.alpha = alpha
        self.xy_prob = None

    def train(self, x, y):
        """
        input:
        x: [[string]] or [[int]], y: array [int]
        output:
        xy_prob: [[float]] nclasses x nwords, y_prob: [float]
        """
        n = np.max(y) + 1
        y_prob = np.ones((n, 1)) * self.alpha
        for c in y:
            y_prob[c] += 1
        y_prob /= np.sum(y_prob)

        word_set = set([word for row in x for word in row])
        word_map = {word: index for index, word in enumerate(word_set)}
        m = len(word_set)

        xy_prob = np.ones((n, m)) * self.alpha
        for row, c in zip(x, y):
            for word in row:
                xy_prob[c][word_map[word]] += 1
        xy_prob /= np.sum(xy_prob, axis=1, keepdims=True)

        self.word_map = word_map
        self.xy_prob = xy_prob
        self.y_prob = y_prob
        return xy_prob, y_prob

    def predict(self, x):
        """
        input:
        x: [[string]] or [[int]]
        output:
        probs: array, [[float]]
        """
        n = len(self.y_prob)
        probs = np.array([[np.product([row[self.word_map[word]] for word in words]) for words in x] for row in self.xy_prob])
        #probs /= np.sum(probs, axis=0, keepdims=True)
        return probs

def count_input(x):
    """transforms a list of steps into counts
    inputs:
        x: [[item]], list of list of items where items belongs to Eq
    outputs:
        [[int]], len(x) x n
    """
    word_set = set([word for row in x for word in row])
    word_map = {word: index for index, word in enumerate(word_set)}
    n = len(word_set)
    counts = np.zeros((len(x), n))

    for crow, row in zip(counts, x):
        for word in row:
            crow[word_map[word]] += 1
    return counts

def transform_labels(y):
    """transforms labels to 0..#distinct labels
    inputs:
        y: [item]
    outputs:
        [int]
    """
    y_set = set(y)
    y_map = {c: index for index, c in enumerate(y_set)}
    return array([y_map[c] for c in y])

def main():
    x = np.array([[1, 2, 1, 3], [1, 2, 2, 2, 3]])
    y = np.array([0, 1])
    clf = NBMultinoulli(alpha=1)
    print(clf.train(x,y))
    print(clf.train_sk(x, y))
    print(clf.predict([[1, 2], [2, 2], [3]]))

if __name__ == '__main__':
    main()
