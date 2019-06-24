import pytest
from utils.accuracy_function import mse

def test_mse_should_return_1():
    y = [2,2,2]
    y_pred = [1,1,1]
    assert mse(y, y_pred) == 1

def test_mse_should_raise_exception_on_y_and_y_predict_with_different_sizes():
    y = [2,1,1]
    y_pred = [1,2]
    with pytest.raises(Exception):
        assert mse(y, y_pred)