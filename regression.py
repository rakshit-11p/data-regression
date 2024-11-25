import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random

df = pd.read_csv("user_behavior_dataset.csv")
df.dropna()
dummy = pd.get_dummies(df, columns=['Operating System'], dtype="int64") # classification based on operating system
data = dummy.drop(columns=['User ID'])
print(data)

x = data[['Age']]
y = data['User Behavior Class']
sns.scatterplot(data=data)
plt.plot(x, y)
plt.show()

# Regression
random.seed(1)
# based on Age, App Usage Time (min/day), Operating System and Battery Drain
x = data[['Age', 'App Usage Time (min/day)', 'Battery Drain (mAh/day)', 'Operating System_Android', 'Operating System_iOS']]
y = data['User Behavior Class']

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression = LinearRegression()
regression.fit(xtrain, ytrain)
print(f"regression score (1) : {regression.score(xtest, ytest)}")

# based on Age and App Usage Time
x = data[['Age', 'App Usage Time (min/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression.fit(xtrain, ytrain)
print(f"regression score(2) : {regression.score(xtest, ytest)}")

# based on Screen On Time (housrs/day), Battery Drain (mAh/day), Data Usage (MB/day)
x = data[['Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression.fit(xtrain, ytrain)
print(f"regression score(3) : {regression.score(xtest, ytest)}")

# based on Battery Drain (mAh/day), Number of Apps Installed
x = data[['Battery Drain (mAh/day)', 'Number of Apps Installed']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression.fit(xtrain, ytrain)
print(f"regression score(4) : {regression.score(xtest, ytest)}")

# based on Age and Data Usage (MB/day)
x = data[['Age', 'Data Usage (MB/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
regression.fit(xtrain, ytrain)
print(f"regression score (5) : {regression.score(xtest, ytest)}")

# based on all (numeric data(s))
x = data[
    ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 
     'Number of Apps Installed', 'Data Usage (MB/day)', 'Age']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
regression.fit(xtrain, ytrain)
print(f"regression score (6) : {regression.score(xtest, ytest)}")
