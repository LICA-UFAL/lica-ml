from knn.knn import KNNClassifier
from examples.datasets import load_numpy_dataset
from utils.accuracy_function import mse

def knn_classifier_iris():
    dataset = load_numpy_dataset("iris")
    trainX, testX, trainY, testY = dataset.train_test_split(0.75, random_seed=0)
    knn = KNNClassifier(5)
    knn.fit(trainX, trainY)

    print("mse {0}".format(mse(knn.predict(testX), testY)))

