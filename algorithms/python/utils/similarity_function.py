import numpy as np 

def euclidian_dist(first_el, second_el):
    return np.sum((np.array(first_el) - np.array(second_el)) ** 2 ) ** 0.5

def manhattan_dist(first_el, second_el):
    return sum(abs(np.array(first_el) - np.array(second_el)))
