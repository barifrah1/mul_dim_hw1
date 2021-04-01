import numpy as np
import pandas as pd
from Consts import *


class Args():

    def __init__(self, *args):
        super(Args, self).__init__(*args)
        self.path = PATH
        self.yIndex = Y_INDEX
        self.xIndices = list(range(1, NUM_OF_COLUMNS + 1))
        self.xIndices.remove(self.yIndex)
        self.xIndices = np.array(self.xIndices)
        self.ridgeLamda = RIDGE_LAMBDA
        self.ridgeLamda2 = RIDGE_LAMBDA2
        self.ridgeLamda3 = RIDGE_LAMBDA3
        self.testSize = TEST_SIZE
        self.lassoLamdas = [1e-8, 1e-5, 1e-4,
                            1e-3, 1e-2, 1e-1, 1, 5, 10, 100]
