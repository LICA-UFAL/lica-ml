import pytest
from utils.train_test_manipulation import train_test_split


def train_test_split_should_raise_exception_not_passing_train_test_size():
    with pytest.raises(Exception):
        assert train_test_split([1,2,3,4])

def train_test_split_should_raise_exception_with_train_test_not_sum_1():
    with pytest.raises(Exception):
        assert train_test_split([1,2,3,4,5], train_size = 0.6, test_size = 0.9)

def train_test_results_should_by_equal_passing_the_same_random_seed():
    a = [1,2,3,4,5,6]
    b = [1,2,3,4,5,6]
    assert train_test_split(a,b, train_size=0.5, random_seed=0) == train_test_split(a,b, train_size=0.5, random_seed=0)