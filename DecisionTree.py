import numpy as np
import pandas as pd
import warnings
import sys
import sklearn.tree as tree
from sklearn.tree import DecisionTreeClassifier

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
print(x[0:5])
