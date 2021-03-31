import numpy as np
import pandas as pd

from Args import Args
from Dataset import Dataset
from Regression import Regression, plotFigrueOfRegressionCoefficients
from RidgeRegression import RidgeRegression
if __name__ == "__main__":
    args = Args()
    dataset = Dataset(path=args.path, xIndices=args.xIndices,
                      yIndex=args.yIndex)
    print(dataset.yIndex)
    print(dataset.xIndices)
    print(dataset.X.shape)
    subsets = dataset.subsets
    basicRegressionA = Regression(subsets[0][0], subsets[0][1])
    basicRegressionB = Regression(subsets[1][0], subsets[1][1])
    basicRegressionC = Regression(subsets[2][0], subsets[2][1])
    print(basicRegressionA.w, basicRegressionB.w, basicRegressionC.w)
    plotFigrueOfRegressionCoefficients(
        basicRegressionA.w, basicRegressionB.w, basicRegressionC.w)

    # lamda1
    ridgeA = RidgeRegression(subsets[0][0], subsets[0][1], args.ridgeLamda)
    ridgeB = RidgeRegression(subsets[1][0], subsets[1][1], args.ridgeLamda)
    ridgeC = RidgeRegression(subsets[2][0], subsets[2][1], args.ridgeLamda)
    plotFigrueOfRegressionCoefficients(
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = 1")
    # lamda2
    ridgeA = RidgeRegression(subsets[0][0], subsets[0][1], args.ridgeLamda2)
    ridgeB = RidgeRegression(subsets[1][0], subsets[1][1], args.ridgeLamda2)
    ridgeC = RidgeRegression(subsets[2][0], subsets[2][1], args.ridgeLamda2)
    plotFigrueOfRegressionCoefficients(
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = 1e-2")
    # lamda3
    ridgeA = RidgeRegression(subsets[0][0], subsets[0][1], args.ridgeLamda3)
    ridgeB = RidgeRegression(subsets[1][0], subsets[1][1], args.ridgeLamda3)
    ridgeC = RidgeRegression(subsets[2][0], subsets[2][1], args.ridgeLamda3)
    plotFigrueOfRegressionCoefficients(
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = 1e-10")
