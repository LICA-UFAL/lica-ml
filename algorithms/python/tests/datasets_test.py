import pytest
import pandas as pd
from examples.datasets import Dataset, load_numpy_dataset

def test_should_load_iris_dataset():
    assert load_numpy_dataset("iris")

def test_should_raise_exception_to_create_dataset_without_dataframe_or_dataXY():
    with pytest.raises(Exception):
        Dataset()

def test_should_raise_exception_to_train_test_split_dataset_without_set_dataXY_or_pass_dataXY_columns():
    dataframe = pd.DataFrame(data = [[1,2],[1,2],[2,3]])
    with pytest.raises(Exception):
        Dataset(dataframe = dataframe).train_test_split(0.5)