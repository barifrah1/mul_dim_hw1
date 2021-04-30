import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import cross_val_score
from Main_3_1 import myPca


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

    # Split the data into train and test with 80 train / 20 test
    X = data.drop(["Class"], axis=1)
    Y = data["Class"]

    accuracy = []
    X_train_test, X_val, y_train_test, y_val = train_test_split(
        X, Y, train_size=0.7347, random_state=0)  # returns 10000 tuples for train+test and the rest is validation
    for i in np.arange(1, 16):
        Mean_col, Std_col, X_normalized, eigenvalues, eigenvectors, X_projected = myPca(
            X_train_test, i)

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
