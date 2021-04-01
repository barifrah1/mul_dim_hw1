import numpy as np
import pandas as pd

from Args import Args
from Dataset import Dataset
from Regression import Regression, plotFigrueOfRegressionCoefficients
from RidgeRegression import RidgeRegression
from LassoRegression import LassoRegression, plotMSEbyLamda
if __name__ == "__main__":
    args = Args()
    dataset = Dataset(path=args.path, xIndices=args.xIndices,
                      yIndex=args.yIndex, testSize=args.testSize)
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
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = "+str(args.ridgeLamda))
    # lamda2
    ridgeA = RidgeRegression(subsets[0][0], subsets[0][1], args.ridgeLamda2)
    ridgeB = RidgeRegression(subsets[1][0], subsets[1][1], args.ridgeLamda2)
    ridgeC = RidgeRegression(subsets[2][0], subsets[2][1], args.ridgeLamda2)
    plotFigrueOfRegressionCoefficients(
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = "+str(args.ridgeLamda2))
    # lamda3
    ridgeA = RidgeRegression(subsets[0][0], subsets[0][1], args.ridgeLamda3)
    ridgeB = RidgeRegression(subsets[1][0], subsets[1][1], args.ridgeLamda3)
    ridgeC = RidgeRegression(subsets[2][0], subsets[2][1], args.ridgeLamda3)
    plotFigrueOfRegressionCoefficients(
        ridgeA.w, ridgeB.w, ridgeC.w, titleAddition="with lamda = "+str(args.ridgeLamda3))

    #best_lamda is lamda2

    # Lasso
    lamdasForLasso = args.lassoLamdas
    lassoResults = []
    mseList = []
    for lamda in lamdasForLasso:
        lassoReg = LassoRegression(dataset.X_train, dataset.y_train, lamda)
        yPredict = lassoReg.predict(dataset.X_test)
        mse = lassoReg.calculateMSE(dataset.y_test, yPredict)
        mseList.append(mse)
        lassoResults.append((lamda, lassoReg.w, mse))
        total = 0
        for elem in lassoReg.w:
            if(elem != 0):
                total += 1
        print(f"for lamda {lamda} - numer of parameters!=0 : {total}")
    plotMSEbyLamda(lamdasForLasso, mseList)
    print(lassoResults[5])
