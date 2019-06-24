def mse(y, y_pred):
    if len(y) != len(y_pred):
        raise Exception("y and y_pred must be the same size")
    return sum(list(map(lambda eleY, eleY_pred : (eleY - eleY_pred)**2, y, y_pred))) / len(y)

