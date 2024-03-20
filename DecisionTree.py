import sys
import warnings
import numpy as np
import pandas as pd
import sklearn.tree as tree
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
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
# future note: i maded a mistake, now i need to remove column drug from this array
x = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values


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

# modeling
drug_tree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drug_tree.fit(x_trainset, y_trainset)
pred_tree = drug_tree.predict(x_testset)

# checking accuracy, close to 1.0 = better
print('DecisionTree accuracy: ', metrics.accuracy_score(y_testset, pred_tree))

# visualization
export_graphviz(
    drug_tree,
    out_file='tree.dot',
    filled=True,
    feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'],
)
