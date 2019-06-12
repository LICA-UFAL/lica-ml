import pytest
from knn.knn import KNNClassifier

def create_default_knnClassifier():
    n_neighbors = 3
    return KNNClassifier(n_neighbors = n_neighbors)

def test_should_create_KNNClassifier():
    n_neighbors = 10
    assert KNNClassifier(n_neighbors = n_neighbors)

def test_should_fit_knn_classifier():
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [1, 1, 0, 0]
    create_default_knnClassifier().fit(x_train, y_train)

def test_should_predict_sample_label_0():
    x_train = [[1, 1], [3, 1], [1,2], [1,2]]
    y_train = [0, 0, 0, 1]
    x_sample = [1, 2]

    knn_classifier = create_default_knnClassifier()
    knn_classifier.fit(x_train, y_train)
    assert knn_classifier.predict_sample(x_sample) == 0

def test_should_raise_ValueError_on_fit_knn_classifier_with_x_and_y_off_different_sizes():
    x_train = [[1, 1], [2, 1], [3, 2], [4, 2]]
    y_train = [1, 1, 0]

    with pytest.raises(ValueError):
        assert create_default_knnClassifier().fit(x_train, y_train)

def test_should_raise_exception_on_fit_knn_classifier_with_x_small_than_n_neighbords():
    x_train = [[1, 1], [2, 1]]
    y_train = [1,0]

    with pytest.raises(Exception):
        assert create_default_knnClassifier().fit(x_train, y_train).fit(x_train, y_train)

def test_should_raise_ValueError_on_predict_knn_classifier_without_fit():
    x_test = [[1,2],[3,4],[7,8]]

    with pytest.raises(ValueError):
        create_default_knnClassifier().predict(x_test)
