import numpy as np

class NBMultinoulli:
    """
    plain multinoulli NB
    """
    def init(self, dtype=int):
        self.dtype = dtype

    def train(self, x, y):
        """
        input:
        x: [[string]] or [[int]], y: array [int]
        output:
        xy_prob: [[float]] nclasses x nwords, y_prob: [float]
        """
        n = np.max(y) + 1
        y_prob = np.zeros((n, 1))
        for c in y:
            y_prob[c] += 1
        y_prob /= np.sum(y_prob)

        word_set = set([word for row in x for word in row])
        word_map = {word: index for index, word in enumerate(word_set)}
        m = len(word_set)

        xy_prob = np.zeros((n, m))
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
        probs /= np.sum(probs, axis=0, keepdims=True)
        return probs

def main():
    x = np.array([[1, 2, 1, 3], [1, 2, 2, 2, 3]])
    y = np.array([0, 1])
    clf = NBMultinoulli()
    clf.train(x, y)
    print(clf.predict([[1, 2], [2, 2], [3]]))

if __name__ == '__main__':
    main()
