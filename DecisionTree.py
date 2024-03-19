import numpy as np
import pandas as pd
import warnings
import sys
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


# supress warnings:
def warn(*args, **kwargs):
    pass


warnings.warn = warn

# importing dataframe
df = pd.read_csv('drug200.csv', delimiter=",")

# checking how many rows
# print(my_data.shape)

# Defining the matrix and checking
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']].values


# SKlearnig decision tree does not handle categorical variables, cause of that, we can still convert these features to numerical
# For example, let's assign the value 0 to "F" and 1 to "M". In the scenario where there is a third option, it will be represented by the number 2.

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:, 1] = le_sex.transform(x[:, 1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_bp.transform(x[:, 2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
x[:, 3] = le_chol.transform(x[:, 3])

# filling the target variable
y = df['Drug']

# splitting data for test and training
x_trainset, x_testset, y_trainset, y_testset = train_test_split(
    x, y, test_size=0.3, random_state=3
)


# checking if the dimensions match
print(
    f'Shape of X training set {x_trainset.shape}',
    '&',
    f' Size of Y training set {y_trainset.shape}',
)

print(
    f'Shape of X training set {x_testset.shape}',
    '&',
    f' Size of Y training set {y_testset.shape}',
)
