import numpy as np 

def euclidian_dist(first_el, second_el):
    return np.sum((np.array(first_el) - np.array(second_el)) ** 2 ) ** 0.5

def manhattan_dist(first_el, second_el):
    return sum(abs(np.array(first_el) - np.array(second_el)))

def cosine_dist(first_el, second_el):

    first_el = np.array(first_el)
    second_el = np.array(second_el)

    def norm(arr):
        return np.sqrt(np.sum(arr ** 2))

    return abs((np.sum(first_el * second_el)) / (norm(first_el) * norm(second_el)) - 1)