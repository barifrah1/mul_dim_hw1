import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from Regression import Regression
from matplotlib import pyplot as plt


class LassoRegression(Regression):
    def __init__(self, X: np.array, y: np.array, lamda: float):
        self.lamda = lamda
        super(LassoRegression, self).__init__(X, y)

    def calculateCoefficients(self):
        self.clf = linear_model.Lasso(alpha=self.lamda)
        self.clf.fit(self.X, self.y)
        self.w = self.clf.coef_

    def predict(self, X):
        return self.clf.predict(X)

    def calculateMSE(self, yTrue, yPredict):
        return mean_squared_error(yTrue, yPredict)


def plotMSEbyLamda(lamdas: list, mse: list):
    plt.plot(lamdas, mse, color='red', linestyle='--')
    plt.xscale("log")
    plt.xlabel("lamda(log scale)")
    plt.ylabel("MSE")
    plt.title("MSE by lamda for Lasso Regression")
    plt.show()
