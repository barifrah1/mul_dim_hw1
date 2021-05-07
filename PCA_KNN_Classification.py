import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import cross_val_score
from Main_3_1 import myPca
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score


if __name__ == "__main__":
    # Use pandas excel reader
    data_file = './DryBeanDataset/Dry_Bean_Dataset.xlsx'
    data = pd.read_excel(data_file, sheet_name='Dry_Beans_Dataset')

    # scramble the order of the database
    data = data.sample(frac=1).reset_index(drop=True)
    # Change lables to numeric values
    data[['Class']] = data[['Class']].apply(
        lambda col: pd.Categorical(col).codes)

    data['Class'].unique()

    plt.figure(figsize=(12, 6))
    sns.heatmap(data.corr(), annot=True)

    plt.figure(figsize=(12, 6))
    plt.title('Class Count')
    sns.set_style('white')
    sns.countplot(x='Class', data=data)
    plt.show()

    #Divide the data to input features and output
    X = data.drop(["Class"], axis=1)
    Y = data["Class"]

    accuracy = []
    #Split the data into 10000 train+test & validation of 3611
    X_train_test, X_val, y_train_test, y_val = train_test_split(
        X, Y, train_size=0.7347, random_state=0)  # returns 10000 tuples for train+test and the rest is validation
    for i in np.arange(1, 16):
        Mean_col, Std_col, X_normalized, eigenvalues, eigenvectors, X_projected = myPca(
            X_train_test.values, i)

        model = KNeighborsClassifier(n_neighbors=3)
        # StratifiedKFold CV is being done
        scores = cross_val_score(
            model, X_projected, y_train_test, scoring="accuracy", cv=5)
        # we'll store the average value of accuracy calculated over all the 5 CV sets
        accuracy.append(scores.mean())

    plt.figure(figsize=(12, 3))
    accuracy = np.array(accuracy)
    plt.plot(np.arange(1, 16), 1-accuracy)
    plt.xlabel("num of PC's")
    plt.ylabel("1 - accuracy")
    plt.xticks(np.arange(1, 16))
    plt.legend()
    plt.show()

    max_d_accuracy = np.argmax(accuracy)

    max_d_Mean_col, max_d_Std_col, max_d_X_normalized, max_d_eigenvalues, max_d_eigenvectors, max_d_X_projected = myPca(
        X_train_test.values, max_d_accuracy+1)

    #To show that these eigenvectors are orthogonal, we will compute their dot product to show that!
    np.dot(max_d_eigenvectors[:, 0], max_d_eigenvectors[:, 1])

    if max_d_accuracy > 3:
        print(f'd equals {max_d_accuracy}, plotting the projected data to the first 3 PCs')
        # Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(max_d_X_projected[:,0:3], columns=['PC1', 'PC2', 'PC3']).reset_index(drop=True)
        y_train_test_df = pd.DataFrame(y_train_test, columns=['Class']).reset_index(drop=True)
        # Concat it with target variable to create a complete Dataset
        projected_data = pd.concat([principal_df, y_train_test_df], axis=1)

        #Plot
        # Plot 3d graph for PC1, PC2 & PC3
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(15, -60)

        scatter = ax.scatter(projected_data['PC1'], projected_data['PC2'], projected_data['PC3'],
                   c=projected_data['Class'], alpha=0.6, label='Class')

        #produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Class")
        ax.add_artist(legend1)

        # chart
        plt.title("Data projected on 3 PCs")
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        plt.show()

    else:
        print(f'd equals:{max_d_accuracy}, plotting the projected data to the first {max_d_accuracy} PCs')
        # Creating a Pandas DataFrame of reduced Dataset
        principal_df = pd.DataFrame(max_d_X_projected[:,0:2], columns=['PC1', 'PC2']).reset_index(drop=True)
        y_train_test_df = pd.DataFrame(y_train_test, columns=['Class']).reset_index(drop=True)
        # Concat it with target variable to create a complete Dataset
        projected_data = pd.concat([principal_df, y_train_test_df], axis=1)

        # Plot the 2d PC's
        fig, ax = plt.subplots()
        scatter = ax.scatter(projected_data['PC1'], projected_data['PC2'], c=projected_data['Class'])
        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements(),
                            loc="lower left", title="Classes")
        ax.add_artist(legend1)
        plt.show()

    #Project the validation set into the top d PCs
    X_val_normalized = StandardScaler().fit_transform(X_val)
    # Get the new projected data  pcaData = normalizedData * projectionVectors
    X_val_projected = np.dot(max_d_eigenvectors.transpose(),X_val_normalized.transpose()).transpose()

    model = KNeighborsClassifier(n_neighbors=3)
    # StratifiedKFold CV is being done
    scores = cross_val_score(
        model, X_val_projected, y_val, scoring="accuracy", cv=5)
    # we'll store the average value of accuracy calculated over all the 5 CV sets
    accuracy_val= scores.mean()

    #Plot the train and the validation toghether
    # Plot 3d graph for PC1, PC2 & PC3
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(15, -60)

    scatter = ax.scatter(projected_data['PC1'], projected_data['PC2'], projected_data['PC3'],
                         c=projected_data['Class'], alpha=0.6, label='Class')

    # produce a legend with the unique colors from the scatter
    legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Class")
    ax.add_artist(legend1)

    # chart
    plt.title("Data projected on 3 PCs")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    #Add the validation data points in black
    scatter = ax.scatter(X_val_projected[:,0], X_val_projected[:,1], X_val_projected[:,2], c='black', alpha=0.6)
    legend2 = ax.legend(*scatter.legend_elements(), title="Validation")
    ax.add_artist(legend2)

    plt.show()

    ##Plot the Eigenvalues of the constructed PCA
    x_pos = np.arange(len(eigenvalues))
    # Create bars
    plt.bar(x_pos, eigenvalues)
    #Create names on the x-axis
    plt.xticks(x_pos, x_pos+1)
    # Show graphic
    plt.title('Plot Eigenvalues')
    plt.show()

    #Calculate the explained variation
    eig_vals_total = np.sum(eigenvalues)
    var_exp = eigenvalues / eig_vals_total
    cum_var_exp = np.cumsum(var_exp)
    print('Variance Explained: ', var_exp)
    print('Cumulative Variance Explained: ', cum_var_exp)

    #Plot the explained variation
    tmp = np.arange(len(eigenvalues))+1
    plt.bar(tmp, var_exp, alpha=0.5,
            align='center', label='individual explained variance')
    plt.step(tmp, cum_var_exp,
             where='mid', label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xticks(tmp)
    plt.yticks(np.arange(1,step=0.1))
    plt.xlabel('Principal components')
    plt.ylim(0, 1.1)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    # PCA reconstruction=PC scores⋅Eigenvectors.T+Mean
    reconstructed_X = (max_d_X_projected @ max_d_eigenvectors.T) * Std_col + Mean_col
    reconstructed_error = X_train_test,reconstructed_X

    #Calculate reconstruction error
    rmse = sqrt(mean_squared_error(X_train_test, reconstructed_X))
    r2 = r2_score(X_train_test, reconstructed_X)


    #####################################################

    fig = plt.figure(figsize=(10, 8))
    # set width of bars
    barWidth = 0.25

    # Set position of bar on X axis
    r1 = np.arange(len(eigenvectors[:,1]))+1
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    r4 = [x + barWidth for x in r3]

    # Make the plot
    plt.bar(r1, eigenvectors[:,0], color='#99d8c9', width=barWidth, edgecolor='white', label='PC1')
    plt.bar(r2, eigenvectors[:,1], color='#fdbb84', width=barWidth, edgecolor='white', label='PC2')
    plt.bar(r3, eigenvectors[:,2], color='#e7e1ef', width=barWidth, edgecolor='black', label='PC3')
    plt.bar(r4, eigenvectors[:, 3], color='#de2d26', width=barWidth, edgecolor='black', label='PC4')

    # Add xticks on the middle of the group bars
    plt.xticks(np.arange(len(eigenvectors[:,1]))+1)
    plt.xlabel('group', fontweight='bold')

    # Create legend & Show graphic
    plt.legend()
    plt.show()

    #For PC3 the most contributing features are: 10 & 16 with positive contribution and we see that feature 12,14,15 have some negative contibution
    #For PC4 the most contributing features are: 9 with positive contribution and 16 with negative contibution
