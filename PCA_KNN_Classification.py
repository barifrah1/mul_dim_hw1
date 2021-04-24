import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.model_selection import cross_val_score
from Main_3_1 import myPca

# Use pandas excel reader
data_file = './DryBeanDataset/Dry_Bean_Dataset.xlsx'
data = pd.read_excel(data_file, sheet_name='Dry_Beans_Dataset')

#scramble the order of the database
data = data.sample(frac=1).reset_index(drop=True)
#Change lables to numeric values
data[['Class']] = data[['Class']].apply(lambda col:pd.Categorical(col).codes)

data['Class'].unique()

plt.figure(figsize=(12,6))
sns.heatmap(data.corr(),annot=True)

plt.figure(figsize=(12,6))
plt.title('Class Count')
sns.set_style('white')
sns.countplot(x='Class',data=data)


# Split the data into train and test with 80 train / 20 test
X = data.drop(["Class"], axis=1)
Y = data["Class"]

accuracy = []
for i in np.arange(1, 16):
    Mean_col, Std_col, X_meaned, eigenvalues, eigenvectors, X_projected = myPca(X,i)
    X_train, X_test, y_train, y_test = train_test_split(X_projected, Y, train_size=0.75, random_state=0)

    model = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(model, X_train, y_train, scoring="accuracy", cv=5)  # StratifiedKFold CV is being done
    accuracy.append(scores.mean())  # we'll store the average value of accuracy calculated over all the 10 CV sets

plt.figure(figsize=(12,3))
accuracy = np.array(accuracy)
plt.plot(np.arange(1,16), 1-accuracy)
plt.xlabel("n_neighbors")
plt.ylabel("1 - accuracy")
plt.xticks(np.arange(1,16))
plt.legend()