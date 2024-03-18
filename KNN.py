import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

# importing the dataset and printing to see if it worked, df = dataframe
df = pd.read_csv("teleCust1000t.csv")
df.head()
# print(df)

# data Visualization - how many of each class is in your data set
# print(df['custcat'].value_counts())

# print with columns to see what columns we have to use it
# print(df.columns)

# converting the pandas data frame to a numpy array
df_array = df[
    [
        'region',
        'tenure',
        'age',
        'marital',
        'address',
        'income',
        'ed',
        'employ',
        'retire',
        'gender',
        'reside',
        'custcat',
    ]
]
# print(df_array[0:5])

# what are our labels?
y = df['custcat'].values
# print(y[0:5])

# normalizing the data
df_array = (
    preprocessing.StandardScaler()
    .fit(df_array)
    .transform(df_array.astype(float))
)
# print(df_array[0:5])


# Let's split our test data to include an out-of-sample subset, enhancing the accuracy of our
# classification model and guarding against overfitting.
# 20% of our dataset has been set aside for testing purposes, with a random state of 4 ensuring consistent data division.
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    df_array, y, test_size=0.2, random_state=4
)
# print('Train set: ', x_train.shape, y_train.shape)
# print('Test set: ', x_test.shape, y_test.shape)
