import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import random

df = pd.read_csv("user_behavior_dataset.csv")
dummies = pd.get_dummies(df, columns=['Operating System'], dtype='int64')
data = dummies.drop(columns=['User ID'])

sns.scatterplot(data)
plt.show()

y = data['User Behavior Class']

random.seed(1)

# based on Age, Data Usage, Screen On Time, App Usage Time, Battery Drain
x = data[['Age', 'Data Usage (MB/day)', 'Screen On Time (hours/day)', 'App Usage Time (min/day)', 'Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (1) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

# based on Age, Battery Drain
x = data[['Age', 'Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
# n_neighbors = 21
knn = KNeighborsClassifier(n_neighbors=21)
knn.fit(xtrain, ytrain)
print(f"knn regression accuracy (2) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

# based on Battery Drain only
x = data[['Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn.fit(x, y)
print(f"knn regression score (3) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

# changing knn neighbors for the same to 3
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x, y)
print(f"knn regression score (3) (neighbors=3) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")

# based on Screen Time and Age
x = data[['Age', 'Screen On Time (hours/day)']]
# for neighbors = 31
knn = KNeighborsClassifier(n_neighbors=31)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn.fit(x, y)
print(f"knn regression score (4) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}")

# based on screen time, drain, data usage
x = data[['Data Usage (MB/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
knn.fit(x, y)
print(f"knn regression score (5) : {knn.score(xtest, ytest)} or {knn.score(xtest, ytest)*100:.3f}%")
