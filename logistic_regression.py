import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import random


df = pd.read_csv("user_behavior_dataset.csv")
dummy = pd.get_dummies(df, columns=['Operating System'])
data = dummy.drop(columns=['User ID'])

sns.scatterplot(data)
plt.show()

y = data['User Behavior Class']

random.seed(1)

# based on Age, Data Usage, Screen On Time, App Usage Time, Battery Drain
x = data[['Age', 'Data Usage (MB/day)', 'Screen On Time (hours/day)', 'App Usage Time (min/day)', 'Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
model = LogisticRegression()
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (1) : {model.score(xtrain, ytrain)}")
print(f"accuracy (1) : {accuracy} or {accuracy*100:.3f}%")

# based on Age, Battery Drain
x = data[['Age', 'Battery Drain (mAh/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (2) : {model.score(xtrain, ytrain)}")
print(f"accuracy (2) : {accuracy} or {accuracy * 100 : .3f}%")

# based on Screen On Time, Battery Drain and Data Usage
x = data[['Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (3) : {model.score(xtrain, ytrain)}")
print(f"accuracy (3) : {accuracy} or {accuracy*100:.3f}%")

# based on Battery drain and number of apps installed
x = data[['Battery Drain (mAh/day)', 'Number of Apps Installed']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (4) : {model.score(xtrain, ytrain)}")
print(f"accuracy (4) : {accuracy} or {accuracy*100:.3f}%")

# based on Age and Data Usage
x = data[['Age', 'Data Usage (MB/day)']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (5) : {model.score(xtrain, ytrain)}")
print(f"accuracy (5) : {accuracy} or {accuracy*100:.3f}%")

# based on all entries (numeric)
x = data[
    ['App Usage Time (min/day)', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 
     'Number of Apps Installed', 'Data Usage (MB/day)', 'Age']]
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.30)
model.fit(xtrain, ytrain)
ypred = model.predict(xtest)
accuracy = accuracy_score(ytest, ypred)
print(f"Logistic Regression score (6) : {model.score(xtrain, ytrain)}")
print(f"accuracy (6) : {accuracy} or {accuracy*100:.3f}%")
