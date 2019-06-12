import numpy as np
from functools import cmp_to_key
from statistics import mode
from utils.similarity_function import euclidian_dist


class KNNClassifier():
    def __init__(self, n_neighbors, similarity_function = euclidian_dist):
        self.fitted = False
        self.n_neighbors = n_neighbors
        self.similarity_function = similarity_function
        self.data_x = []
        self.data_y = []
    
    def fit(self, x, y):
        if len(x) < self.n_neighbors:
            raise Exception("Input size should by greater than n_neighbors")
        if len(x) != len(y):
            raise ValueError("X and y have different sizes")
        self.data_x = np.array(x)
        self.data_y = np.array(y)
        self.fitted = True

    def predict_sample(self, x):
        x_array = np.array(x)
        if not self.fitted:
            raise ValueError("KNNClassifier not fitted")
        if x_array.shape != self.data_x.shape[1:]:
            raise Exception("sample {0} have different shape than train data".format(x))

        data = [(element, label) for element, label in zip(self.data_x, self.data_y)]
        data = sorted(data, key=cmp_to_key(self._compare(x)))

        return mode(list(map(lambda a: a[1], data[-self.n_neighbors:])))

    def predict(self, x):
        return list(map(self.predict_sample, x))

    def _compare(self, x):
        return lambda a, b: self.similarity_function(x, a[0]) > self.similarity_function(x, b[0])
