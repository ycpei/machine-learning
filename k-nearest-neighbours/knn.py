def knn(k, x, l, y):
    """
    int k; 2d array x: known data; array l: labels of known data; array y: data for prediction
    return: labels of prediction data
    """
    distances = linalg.norm(np.expand_dims(y, axis=1) - np.expand_dims(x, axis=0), axis=2)
    kIndices = np.argpartition(distances, k - 1, axis=1)[:,:k]
    kLabels = np.vectorize(lambda x: l[x])(kIndices)
    return np.vectorize(lambda x: np.unique(x, return_counts=True))(kLabels)
