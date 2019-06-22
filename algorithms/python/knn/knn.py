import numpy as np
from functools import cmp_to_key
from statistics import mode
from utils.similarity_function import euclidian_dist
from utils.standardization_function import standardization


class KNN():
    def __init__(self, n_neighbors, weights=False, similarity_function=euclidian_dist):
        self.fitted = False
        self.n_neighbors = n_neighbors
        self.similarity_function = similarity_function
        self.weights = weights
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

    def predict(self, x):
        return list(map(self.predict_sample, x))

    def predict_sample(self, x_sample):
        raise Exception("base knn class can't predict sample")

    def most_similar_elements(self, x_sample):
        x_array = np.array(x_sample)
        if not self.fitted:
            raise ValueError("KNNClassifier not fitted")
        if x_array.shape != self.data_x.shape[1:]:
            raise Exception("sample {0} have different shape than train data".format(x_sample))

        data = map(lambda element, label: (self.similarity_function(x_sample, element) if self.weights else 1, label),
                   self.data_x, self.data_y)
        data = sorted(data, key=self._compare)
        data = np.array(data)
        data = data[:self.n_neighbors]

        return data
    
    @staticmethod
    def _compare(ele):
        class K:
            def __init__(self, item):
                self.item = item

            def __gt__(self, other):
                if self.item[0] == other.item[0]:
                    return self.item[1] < other.item[1]
                else:
                    return self.item[0] > other.item[0]

        return K(ele)

class KNNClassifier(KNN):
    def __init__(self, n_neighbors, weights=False, similarity_function=euclidian_dist):
        super().__init__(n_neighbors, weights, similarity_function)

    def predict_sample(self, x_sample):
        data = self.most_similar_elements(x_sample)
        data[:,0] = standardization(data[:,0])

        dict_labels = {}
        for similarity, label in data:
            ele = dict_labels.get(label, (0, 0))
            dict_labels[label] = (ele[0] + similarity, ele[1] + 1)

        data = list(map(lambda label, ele: (ele[0]/ele[1], ele[1], label), dict_labels.keys(), dict_labels.values()))
        data = sorted(data, key=self._compare)

        return data[0][2]

class KNNRegressor(KNN):
    def __init__(self, n_neighbors, weights=True, similarity_function=euclidian_dist):
        super().__init__(n_neighbors, weights, similarity_function)
    
    def predict_sample(self, x_sample):
        data = self.most_similar_elements(x_sample)
        data = list(map(lambda a: (1/a[0] if a[0] !=0 else 1, a[1]), data))
        numerator = sum(map(lambda a: a[0] * np.array(a[1]), data))
        denominator = sum(map(lambda a: a[0], data))
        return numerator/denominator

