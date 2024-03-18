import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing

# importing the dataset and printing to see if it worked
df = pd.read_csv("teleCust1000t.csv")
df.head()
print(df)
