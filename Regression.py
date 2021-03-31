import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class Regression():
    def __init__(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.calculateCoefficients()

    def calculateCoefficients(self):
        X_transpose = np.transpose(self.X)
        XtX = np.matmul(X_transpose, self.X)
        self.w = np.matmul(np.matmul(np.linalg.inv(XtX), X_transpose), self.y)
        print(self.w.shape, self.w)


def plotFigrueOfRegressionCoefficients(wA, wB, wC, titleAddition=""):
    x = range(2, 24)

    plt.plot(x, wA[1:], color='red', linestyle='--')
    plt.plot(x, wB[1:], color='blue', linestyle='--')
    plt.plot(x, wC[1:], color='green', linestyle='--')
    plt.xlabel("Index of coefficeint(wi")
    plt.ylabel("Coefficeint size in log scale")
    plt.title("Coefficeints stability for different datasets "+titleAddition)

    plt.show()
