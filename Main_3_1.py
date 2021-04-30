import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
import seaborn as sb
import matplotlib.pyplot as plt


from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


def myPca(X, num_components):
    X_normalized = StandardScaler().fit_transform(X)
    # calculate the mean of each column
    Mean_col = np.mean(X, axis=0)
    Std_col = np.std(X, axis=0)
    # center columns by subtracting column means
    #X_meaned = X - Mean_col
    # calculate covariance matrix
    cov_mat = np.cov(X_normalized, rowvar=False)

    # eigendecomposition of covariance matrix
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    eigenvalue_subset = sorted_eigenvalue[0:num_components]
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # In the assigment requested to compute the spectral decomposition
    D_matrix = eigenvalue_subset*np.identity(num_components)
    A_matrix = eigenvector_subset@D_matrix@eigenvector_subset.T

    # Get the new projected data  pcaData = normalizedData * projectionVectors
    X_projected = np.dot(eigenvector_subset.transpose(),
                         X_normalized.transpose()).transpose()

    return Mean_col, Std_col, X_normalized, eigenvalue_subset, eigenvector_subset, X_projected


if __name__ == "__main__":
    data = load_wine()

    Mean_col, Std_col, X_normalized, eigenvalues, eigenvectors, X_projected = myPca(
        data.data, 3)

    # Creating a Pandas DataFrame of reduced Dataset
    principal_df = pd.DataFrame(X_projected, columns=['PC1', 'PC2', 'PC3'])

    # Concat it with target variable to create a complete Dataset
    principal_df = pd.concat([principal_df, pd.DataFrame(
        data.target, columns=['target'])], axis=1)

    # Plot the 2d PC's
    plt.figure(figsize=(6, 6))
    sb.scatterplot(data=principal_df, x='PC1', y='PC2',
                   hue='target', s=60, palette='icefire')
    # plt.show()

    # PCA reconstruction=PC scores⋅Eigenvectors.T+Mean
    reconstructed_X = (X_projected@eigenvectors.T)*Std_col + Mean_col

    # Plot 3d graph for reconstruction plot for column #4,#5 & #6
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, -60)

    recon_class = reconstructed_X
    original_class = data.data
    ax.scatter(recon_class[:, 4], recon_class[:, 5], recon_class[:, 6],
               color='red', alpha=0.6, label='reconstructed data')
    ax.scatter(original_class[:, 4], original_class[:, 5],
               original_class[:, 6], color='green', alpha=0.6, label='Original data')

    # chart
    plt.title("3 columns of original and reconstructed data of Wine Quality")
    ax.set_xlabel('Col#4')
    ax.set_ylabel('Col#5')
    ax.set_zlabel('Col#6')
    plt.legend()
    plt.show()
