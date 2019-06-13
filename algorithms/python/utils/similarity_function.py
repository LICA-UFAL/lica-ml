import numpy as np 

def euclidian_dist(first_el, second_el):
    return np.sum((np.array(first_el) - np.array(second_el)) ** 2 ) ** 0.5

def manhattan_dist(first_el, second_el):
    return sum(abs(np.array(first_el) - np.array(second_el)))

def cosine_dist(first_el, second_el):

    def magnitude(arr):
        return np.sqrt(np.sum(arr ** 2))

    return np.sum(first_el * second_el) / (magnitude(first_el) * magnitude(second_el))