import numpy as np

SOURCE_FILE = "source data/iris.data"

def to_array():
    file = open(SOURCE_FILE, "r")
    dataX = []
    dataY = []
    
    for line in list(file.readlines())[:-1]:
        array = line.split(",")
        dataX.append(list(map(lambda a: float(a), array[:-1])))
        dataY.append(array[-1])

    dict_map = {value:index for index, value in enumerate(set(dataY))}
    dataY = list(map(lambda ele: dict_map[ele], dataY))

    return dataX, dataY

def to_numpy_file():
    dataX, dataY = to_array()
    np.save("dataX", dataX)
    np.save("dataY", dataY)

if __name__ == "__main__":
    to_numpy_file()
