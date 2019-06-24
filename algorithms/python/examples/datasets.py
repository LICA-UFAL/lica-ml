import json
import numpy as np
import pandas as pd

from utils import train_test_manipulation

DATASETS_PATH = "../datasets/"


def create_dataframe(data, dataframe_name, columns = None):
    dataframe = pd.DataFrame(data = data, columns = columns)
    if not columns:
        dataframe.columns = ["{0} feature {1}".format(dataframe_name, column) for column 
                            in dataframe.columns]

    return dataframe

class Dataset():
    def __init__(self, dataframe = None, dataX = None, dataY = None, description = {}):
        if dataframe is None and dataX is not None and dataY is not None:
            dataframe_dataX = create_dataframe(dataX, "dataX", description.get("dataX feature(s) name"))
            dataframe_dataY = create_dataframe(dataY, "dataY", description.get("dataY feature(s) name"))
            dataframe = dataframe_dataX.join(dataframe_dataY)
        elif dataframe and not dataX and not dataY:
            if "dataX feature(s) name" in description and "dataY feature(s) name" in description:
                dataX = dataframe.get(description["dataX feature(s) name"])
                dataY = dataframe.get(description["dataY feature(s) name"])
        else:
            raise Exception("Can't get enough data to create dataset")

        self.dataX = dataX
        self.dataY = dataY
        self.dataframe = dataframe
        self.dataset_type = description["problem type"]
        
        self.description = description

    def summary(self):
        self.dataframe.describe()

    def dataXY(dataX_columns, dataY_columns):
        return self.dataframe.get(dataX_columns).values,\
               self.dataframe.get(dataY_columns).values

    def train_test_split(self, train_size, train_columns = None,
                                test_columns = None, random_seed = None):
        if train_columns and test_columns:
            dataX, dataY = self.dataXY(train_columns, test_columns)
        elif self.dataX is not None and self.dataY is not None:
            dataX, dataY = self.dataX, self.dataY
        else:
            raise Exception("can't get dataX and dataY")

        return train_test_manipulation.train_test_split(dataX, dataY, 
                            train_size= train_size, random_seed=random_seed)
             
    
    def __str__(self):
        return str(self.dataframe)

def load_numpy_dataset(dataset_name):
    dataset_folder_path = DATASETS_PATH + dataset_name
    dataX = np.load(dataset_folder_path + "/dataX.npy")
    dataY = np.load(dataset_folder_path + "/dataY.npy")
    description = json.load(open(dataset_folder_path + "/description.txt", "r"))

    return Dataset(dataX = dataX,dataY = dataY, description = description)