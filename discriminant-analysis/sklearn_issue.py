#https://github.com/scikit-learn/scikit-learn/issues/12731
import numpy as np

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
