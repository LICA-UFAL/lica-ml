from knn.knn import KNNClassifier

def create_default_knnClassifier():
    n_neighbors = 10
    return KNNClassifier(n_neighbors = n_neighbors)

def test_should_create_KNNClassifier():
    n_neighbors = 10
    assert KNNClassifier(n_neighbors = n_neighbors)

def test_should_fit_knn_classifier():
    x_train = [[1,1],[2,1],[3,2],[4,2]]
    y_train = [1,1,0,0]
    create_default_knnClassifier().fit(x_train, y_train)

def test_should_raise_ValueError_on_fit_knn_classifier():
    x_train = [[1,1],[2,1],[3,2],[4,2]]
    y_train = [1,1,0]
    create_default_knnClassifier().fit(x_train, y_train)

def test_should_raise_ValueError_on_predict_knn_classifier():
    x_test = [[1,2],[3,4],[7,8]]
    create_default_knnClassifier().predict(x_test)
