import numpy as np

def matrix_sum(matrix_a, matrix_b):
    return np.add(matrix_a, matrix_b)


def matrix_sub(matrix_a, matrix_b):
    return np.subtract(matrix_a, matrix_b)


def matrix_mult(matrix_a, matrix_b):
    matrix_a = np.asmatrix(matrix_a)
    matrix_b = np.asmatrix(matrix_b)
    matrix = matrix_a * matrix_b
    return np.asmatrix(matrix)


def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))


def derivative_sigmoid(x):
    matrix_ = np.asmatrix(x)
    rows = matrix_.shape[0]
    columns = matrix_.shape[1]
    
    matrix = list()
    
    for row in range(rows):
        rows_ = list()
        
        for column in range(columns):
            sigmoid = matrix_[row, column] * (1 - matrix_[row, column])
            rows_.append(sigmoid)
        
        matrix.append(rows_)
        
    return np.asmatrix(matrix)


def scalar_multiply(a, b):
    matrix = np.asmatrix(a)
    matrix *= b
    return matrix


def hadamard(matrix_a, matrix_b):
    matrix_a = np.asmatrix(matrix_a)
    matrix_b = np.asmatrix(matrix_b)
    matrix = np.multiply(matrix_a, matrix_b)
    return matrix
