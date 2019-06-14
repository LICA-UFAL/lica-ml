import pytest
from utils.standardization_function import standardization


def test_standardization_func_1():
    array = [1, 0]

    assert (standardization(array) == [1, 0]).all()


def test_standardization_func_2():
    array = [0, 1]

    assert (standardization(array) == [0, 1]).all()


def test_should_return_array_of_ones_when_max_array_is_equal_to_min_array():
    array = [0, 0, 0, 0]

    assert (standardization(array) == [1, 1, 1, 1]).all()


def test_standardization_should_warns_when_max_array_is_equal_to_min_array():
    array = [0, 0, 0]

    with pytest.warns(Warning):
        standardization(array)
