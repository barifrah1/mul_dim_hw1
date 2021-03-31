import numpy as np
import pandas as pd
from Regression import Regression
from matplotlib import pyplot as plt


class RidgeRegression(Regression):
    def __init__(self, X: np.array, y: np.array, lamda: float):
        self.lamda = lamda
        super(RidgeRegression, self).__init__(X, y)

    def calculateCoefficients(self):
        X_transpose = np.transpose(self.X)
        XtX = np.matmul(X_transpose, self.X)
        regularizationTerm = self.lamda * np.eye(len(XtX))
        XtX += regularizationTerm
        self.w = np.matmul(np.matmul(np.linalg.inv(XtX), X_transpose), self.y)
        print(self.w.shape, self.w)
