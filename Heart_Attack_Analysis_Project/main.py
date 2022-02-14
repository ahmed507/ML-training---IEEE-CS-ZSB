# Heart Attack Analysis & Prediction Using LogisticRegression and SVM
# https://www.kaggle.com/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# import the data
HeatData = pd.read_csv("heart.csv")
# print(HeatData)
HeatData.drop_duplicates(keep='first', inplace=True)
X = HeatData.iloc[:, :-1]
y = HeatData.iloc[:, -1]
# print(X.shape)
# print(y.shape)
# print(HeatData.shape)
# print(X.columns)

scaler = StandardScaler()
# Splitting the data (20% ==>> test and 80% ==>> train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# print(x.shape)
# print(X_test.shape)
# print(X_train.shape)
print("             LogisticRegression")
LogisticRegressionModel = LogisticRegression(solver="liblinear", random_state=2, C=0.01, max_iter=1000, penalty="l2")
LogisticRegressionModel.fit(X_train, y_train)
# print("LogisticRegressionModel train score : {:.2f}%: ".format(LogisticRegressionModel.score(X_train, y_train) * 100))
# print("LogisticRegressionModel test score : {:.2f}%: ".format(LogisticRegressionModel.score(X_test, y_test) * 100))
# print("LogisticRegressionModel Classes : ", LogisticRegressionModel.classes_)
# print("LogisticRegressionModel Number of iterations : ", LogisticRegressionModel.n_iter_)

y_pred = LogisticRegressionModel.predict(X_test)
y_pred_prob = LogisticRegressionModel.predict_proba(X_test)
# print("True : ", numpy.array(y_test[:10]))
# print("pred : ", y_pred[:10])
CM = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(CM)
Acc = accuracy_score(y_test, y_pred)
print("accuracy of LogisticRegression : {:.2f}%: ".format(Acc * 100))

print("======================")
print("         SVM")
SVMModel = SVC(C=1, max_iter=1000, random_state=41)
SVMModel.fit(X_train, y_train)
y_pred = SVMModel.predict(X_test)
print("The accuracy of SVM is : {:.2f}%: ".format(accuracy_score(y_test, y_pred) * 100))
