import pytest
from utils.similarity_function import euclidian_dist

def test_zero_euclidian_dist():
    first_ele = [1, 2, 3, 4]
    second_ele = [1, 2, 3, 4]

    assert euclidian_dist(first_ele, second_ele) == 0

def test_euclidian_dist_should_return_3():
    fist_ele = [1, 1, 1]
    second_ele = [2, 3, 3]

    assert  euclidian_dist(fist_ele, second_ele) == 3

def test_euclidian_dist_should_raise_ValueError():
    first_ele = [1, 2, 3]
    second_ele = [1, 2, 3, 4]

    with pytest.raises(ValueError):
        assert euclidian_dist(first_ele, second_ele)


