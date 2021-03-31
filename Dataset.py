import numpy as np
import pandas as pd
import random as random
from Consts import SUBSET_SIZE, NUM_OF_SUBSETS


class Dataset():
    def __init__(self, *args, **keywords):
        self.data = pd.read_csv(keywords["path"])
        self.columns = self.data._reindex_columns
        data_as_array = self.data.to_numpy()
        self.yIndex = keywords["yIndex"] - 1
        self.xIndices = keywords["xIndices"] - 1
        self.X = data_as_array[:, self.xIndices]
        self.y = data_as_array[:, self.yIndex]
        print(np.ones((len(self.X), 1)).shape, self.X.shape)
        # create w0 with shape num_of_rows X 1
        w0 = np.ones((len(self.X), 1))
        # Concatenate between w0 and X to matrix of size 88 X 17
        self.X = np.concatenate((w0, self.X), axis=1)
        # create 6 corelated columns as last columns
        self.addedColumns = np.zeros((len(self.X), 6))
        self.addedColumns[:, 0] = self.X[:, 1] + self.X[:, 2]
        self.addedColumns[:, 1] = self.X[:, 3] + self.X[:, 4]
        self.addedColumns[:, 2] = self.X[:, 5] + self.X[:, 6]
        self.addedColumns[:, 3] = self.X[:, 7] + self.X[:, 8]
        self.addedColumns[:, 4] = self.X[:, 9] + self.X[:, 10]
        self.addedColumns[:, 5] = self.X[:, 11] + self.X[:, 12]
        self.X = np.concatenate((self.X, self.addedColumns), axis=1)
        # create 3 subsets of the data
        self.subsets = self.createSubsets(NUM_OF_SUBSETS)

    def createSubsets(self, numOfSubsets):
        subsets = []
        for i in range(numOfSubsets):
            randomIndices = np.random.randint(
                low=0, high=len(self.X), size=SUBSET_SIZE)
            subsets.append((self.X[randomIndices, :],
                            self.y[randomIndices]))
        print(subsets[1][0].shape)
        return subsets
