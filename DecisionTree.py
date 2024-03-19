import numpy as np
import pandas as pd
import warnings
import sys
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing


# supress warnings:
def warn(*args, **kwargs):
    pass


warnings.warn = warn

# importing dataframe
df = pd.read_csv('drug200.csv', delimiter=",")
# print(df.head())

# checking how many rows
# print(my_data.shape)

# Defining the matrix and checking
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K', 'Drug']].values
# print(x[0:5])


# SKlearnig decision tree does not handle categorical variables, cause of that, we can still convert
# these features to numerical
# For example, let's assign the value 0 to "F" and 1 to "M". In the scenario where
# there is a third option, it will be represented by the number 2.
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
x[:, 1] = le_sex.transform(x[:, 1])

le_bp = preprocessing.LabelEncoder()
le_bp.fit(['LOW', 'NORMAL', 'HIGH'])
x[:, 2] = le_bp.transform(x[:, 2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
x[:, 3] = le_chol.transform(x[:, 3])

print(x[0:5])
