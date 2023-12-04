import requests
import csv
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('processed_datas.csv')
X = data.iloc[:, :-1] 
y = data.iloc[:, -1]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)
#model = SVR(kernel='rbf', gamma=0.1, C=10, epsilon=0.1)
model.fit(X_train, y_train)
print('模型训练完成！')


y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred) 


accuracy = accuracy_score(y_test, y_pred)
print('模型准确率为：', accuracy)
print('Classification report:\n', report)
y_pred = model.predict(X_test)


'''mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)
print(y_test)
print(y_pred)
print("均方误差: ", mse)
print("均方根误差: ", rmse)
print("R2得分: ", r2)'''
