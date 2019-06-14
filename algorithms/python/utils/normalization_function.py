import numpy as np


def norm(array):
    array = np.array(array)

    if array.var() == 0:
        raise ZeroDivisionError("Variance zero in data")

    return (array - array.mean())/array.var()
