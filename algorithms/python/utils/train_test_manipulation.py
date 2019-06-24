import numpy as np

def train_test_split(dataX, *others, train_size = None, test_size = None, random_seed = None):
    """ 
    Split train and test data, train_size and test_size must be <= 1
    and if both value are passed train_size + test_size == 1
    """
    if random_seed:
        np.random.seed(random_seed)

    if train_size and not test_size:
        test_size = 1 - train_size
    elif test_size and not train_size:
        train_size = 1 - test_size
    else:
        raise Exception("train and test size are none")
    
    if train_size + test_size != 1:
        raise Exception("Train and test size don't sum 1")

    index_array = list(range(len(dataX)))
    train_index_array = np.random.choice(index_array, int(len(dataX)*train_size), replace = False)
    test_index_array = list(filter(lambda ele: ele not in train_index_array, index_array))

    get_train_test = lambda array: [np.array(array)[train_index_array], np.array(array)[test_index_array]]
    train_test_data = []
    for data in [dataX] + list(others):
        train_test_data = train_test_data + get_train_test(data)

    return train_test_data


       