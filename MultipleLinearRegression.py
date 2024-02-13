import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pl
from sklearn import linear_model

df = pd.read_csv("FuelCOnsumptionCo2.csv")
cdf = df[
    [
        "ENGINESIZE",
        "CYLINDERS",
        "FUELCONSUMPTION_CITY",
        "FUELCONSUMPTION_HWY",
        "FUELCONSUMPTION_COMB",
        "CO2EMISSIONS",
    ]
]
# print(cdf.head(9))


# plot emission values with enginesize
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# making a mask to train 80% of the data, that's will provide a better out of
# sample and accuracy on your predict model
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# train data distribuition
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

# multiple regression model
regr = linear_model.LinearRegression()
x = np.asanyarray(
    train[
        [
            "ENGINESIZE",
            "CYLINDERS",
            "FUELCONSUMPTION_CITY",
            'FUELCONSUMPTION_HWY',
        ]
    ]
)
y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(x, y)

# coefficients for your regression line
print(f"Coefficients:{regr.coef_}")

# predction part with sklearn
y_hat = regr.predict(
    test[
        [
            'ENGINESIZE',
            'CYLINDERS',
            'FUELCONSUMPTION_CITY',
            'FUELCONSUMPTION_HWY',
        ]
    ]
)
x = np.asanyarray(
    test[
        [
            'ENGINESIZE',
            'CYLINDERS',
            'FUELCONSUMPTION_CITY',
            'FUELCONSUMPTION_HWY',
        ]
    ]
)
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Mean Squared Error(MSE) : %.2f' % np.mean((y_hat - y**2)))

# Explained variance score: 1 is perfect prediction
print('Variance Score: %.2f' % regr.score(x, y))
