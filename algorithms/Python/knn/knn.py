from utils.similarity_function import euclidian_dist


class KNNClassifier():
    def __init__(self):
        self.fitted = False
    
    def fit(self, x, y):
        self.data_x = x
        self.data_y = y 

    def predict(self, x):
        for el in x:
