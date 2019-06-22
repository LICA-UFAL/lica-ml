import pytest
from knn.knn import KNN,KNNClassifier,KNNRegressor

def create_default_knnClassifier():
    n_neighbors = 3
    return KNNClassifier(n_neighbors = n_neighbors, weights = False)

def create_default_knnRegressor():
    n_neighbors = 3
    return KNNRegressor(n_neighbors = n_neighbors, weights = False)

# KNN tests

def test_should_not_predict_knn():
    knn = KNN(3)
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [[1], [1], [0], [0]]
    x_sample = [1, 1]

    knn.fit(x_train, y_train)
    with pytest.raises(Exception):
        assert knn.predict_sample(x_sample)

def test_should_raise_ValueError_on_fit_knn_with_x_and_y_off_different_sizes():
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [1, 1, 0]

    with pytest.raises(ValueError):
        assert create_default_knnClassifier().fit(x_train, y_train)

def test_should_raise_exception_on_fit_knn_with_x_small_than_n_neighbords():
    x_train = [[1, 1], [2, 1]]
    y_train = [1,0]

    with pytest.raises(Exception):
        assert create_default_knnClassifier().fit(x_train, y_train).fit(x_train, y_train)

# KNNClassifier tests

def test_should_create_KNNClassifier():
    n_neighbors = 10
    assert KNNClassifier(n_neighbors = n_neighbors)


def test_should_fit_knn_classifier():
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [1, 1, 0, 0]
    create_default_knnClassifier().fit(x_train, y_train)


def test_should_predict_sample_label_0_on_knnclassifier():
    x_train = [[1, 1], [3, 1], [1,2], [1,2]]
    y_train = [0, 0, 0, 1]
    x_sample = [1, 2]

    knn_classifier = create_default_knnClassifier()
    knn_classifier.fit(x_train, y_train)
    assert knn_classifier.predict_sample(x_sample) == 0


def test_should_raise_ValueError_on_predict_knn_classifier_without_fit():
    x_test = [[1,2],[3,4],[7,8]]

    with pytest.raises(ValueError):
        create_default_knnClassifier().predict(x_test)

# KNNRegressor tests

def test_should_create_KNNRegressor():
    n_neighbors = 10
    assert KNNRegressor(n_neighbors = n_neighbors)


def test_should_fit_knn_regressor():
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [[1], [1], [0], [0]]
    create_default_knnRegressor().fit(x_train, y_train)

def test_should_predict_1_on_knnregressor():
    x_train = [[1, 1], [3, 1], [1,2], [1,2]]
    y_train = [1, 0, 1, 1]
    x_sample = [1, 2]

    knn_regressor = create_default_knnRegressor()
    knn_regressor.fit(x_train, y_train)
    assert knn_regressor.predict_sample(x_sample) == 1

def test_should_raise_ValueError_on_predict_knn_regressor_without_fit():
    x_test = [[1,2],[3,4],[7,8]]

    with pytest.raises(ValueError):
        create_default_knnRegressor().predict(x_test)
