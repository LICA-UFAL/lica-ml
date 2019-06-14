import warnings
import numpy as np


def standardization(array):
    array = np.array(array)
    if array.max() == array.min():
        warnings.warn("Array max in equal to array min", Warning)
        return np.ones(len(array))
    else:
        return np.array(
            list(map(lambda value: round(value, 4), (array - array.min())/ (array.max() - array.min()))))
