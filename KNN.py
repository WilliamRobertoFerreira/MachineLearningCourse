import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics


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
x_train, x_test, y_train, y_test = train_test_split(
    df_array, y, test_size=0.2, random_state=4
)
# print('Train set: ', x_train.shape, y_train.shape)
# print('Test set: ', x_test.shape, y_test.shape)

# train model and predict
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(x_train, y_train)
# print(neigh)

# predicting
yhat = neigh.predict(x_test)
# print(yhat)

# accuracy Evaluation
# print(
#     'Train set accuracy: ',
#     metrics.accuracy_score(y_train, neigh.predict(x_train)),
# )
# print('test set Accuracy: ', metrics.accuracy_score(y_test, yhat))


# what about the other K?
# trying to fing the best K for our classification model
''' This script calculates the average accuracies for different values of k (number of neighbors)
in the k-nearest neighbors model and stores them in an array, along with the associated 
standard deviations for each accuracy. This allows for an evaluation of the model's 
performance across different k configurations.'''
ks = 10
mean_acc = np.zeros((ks - 1))
std_acc = np.zeros((ks - 1))

for n in range(1, ks):
    # train model and predict
    neigh = KNeighborsClassifier(n_neighbors=n).fit(x_train, y_train)
    yhat = neigh.predict(x_test)
    mean_acc[n - 1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n - 1] = np.std(yhat == y_test) / np.sqrt(yhat.shape[0])
# print(mean_acc)


# Plot the model accuracy for a different number of neighbors
'''
Generates a plot that shows the average accuracy of the k-nearest neighbors model, 
along with one and three standard deviation ranges around that mean, 
allowing visualization of the confidence interval of the model's performance 
for different values of k.'''
plt.plot(range(1, ks), mean_acc, 'g')
plt.fill_between(
    range(1, ks), mean_acc - 1 * std_acc, mean_acc + 1 * std_acc, alpha=0.10
)
plt.fill_between(
    range(1, ks),
    mean_acc - 3 * std_acc,
    mean_acc + 3 * std_acc,
    alpha=0.10,
    color="green",
)
plt.legend(('Accuracy ', '+/- 1xstd', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print(
    "The best accuracy was with",
    mean_acc.max(),
    "with k=",
    mean_acc.argmax() + 1,
)
