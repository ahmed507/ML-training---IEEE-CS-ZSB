import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

'''
        THESE VAlUES HAVE BEEN CHANGED WITH INDEX OF THEIR ARRAY 
# ['Unknown', 'formerly smoked', 'never smoked', 'smokes']
# ['Female', 'Male', 'Other']
# ['No', 'Yes']
# ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children']
# ['Rural', 'Urban']
'''


def sigmoid(theta, x):
    z = x.dot(theta)
    return 1 / (1 + np.exp(-z))


def cost(x, y, theta, m):
    h = sigmoid(theta, x)
    first = -y.T.dot(np.log(h))
    k = np.log(1 - h)
    sec = (1 - y).T.dot(k)
    return (first - sec) / m


def predict(theta, x):
    prob = sigmoid(theta, x)
    return [1 if x >= 0.5 else 0 for x in prob]


def gradiant_descend(x, y, theta, m, alpha, n_iterations):
    for i in range(n_iterations):
        h = sigmoid(theta, x)
        theta -= (np.dot(x.T, h - y)) * (alpha / m)
    return theta


data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.drop("id", inplace=True, axis=1)
data.insert(0, "Ones", 1)
enc = preprocessing.LabelEncoder()
data.iloc[:, -2] = enc.fit_transform(data.iloc[:, -2])
data.iloc[:, 1] = enc.fit_transform(data.iloc[:, 1])
data.iloc[:, 5] = enc.fit_transform(data.iloc[:, 5])
data.iloc[:, 6] = enc.fit_transform(data.iloc[:, 6])
data.iloc[:, 7] = enc.fit_transform(data.iloc[:, 7])
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
data.bmi = imputer.fit_transform(data.bmi.values.reshape(-1, 1)[:])

train_col = data.shape[0] // 2
x_train = np.array(data.iloc[0:train_col, 0:-1].values)
x_test = np.array(data.iloc[train_col:, 0:-1].values)

m, n = x_train.shape
y_train = np.array(data.iloc[0:train_col, -1].values).reshape(m, 1)
y_test = np.array(data.iloc[train_col:, -1].values).reshape(len(data.iloc[train_col:, -1]), 1)
alpha = 0.0005

theta = np.zeros(n).reshape(n, 1)
print("Cost Before :", cost(x_train, y_train, theta, m)[0][0])

gradiant_descend(x_train, y_train, theta, m, alpha, 100000)
print("Cost After :", cost(x_train, y_train, theta, m)[0][0])
p = predict(theta, x_test)

correct = 0
for i in range(len(p)):
    if p[i] == y_test[i]:
        correct += 1
accuracy = (correct / len(y_test)) * 100
print('accuracy = {0}%'.format(accuracy))