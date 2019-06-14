import pytest
from utils.normalization_function import norm


def test_normalization_func_1():
    array = [1, 0]

    assert (norm(array) == [2, -2]).all()


def test_normalization_func_2():
    array = [0, 1]

    assert (norm(array) == [-2, 2]).all()


def test_should_raise_ZeroDivisionError_when_array_variance_is_zero():
    array = [0,0,0]

    with pytest.raises(ZeroDivisionError):
        assert norm(array)


