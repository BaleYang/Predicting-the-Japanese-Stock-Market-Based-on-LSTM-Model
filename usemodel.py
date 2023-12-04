import pandas as pd
import numpy as np

# to plot within notebook
import matplotlib.pyplot as plt
# setting figure size
from matplotlib.pylab import rcParams


# for normalizing data
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

# 加载模型，以后可以直接调用该模型进行预测
model = load_model('model-2.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv('stock_prices.csv')
# print the head
# setting index as date
df['date'] = pd.to_datetime(df.date, format='%Y%m%d')
df.index = df['date']
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0, len(df)), columns=['date', 'close'])
for i in range(0, len(data)):
    new_data['date'][i] = data['date'][i]
    new_data['close'][i] = data['close'][i]

# setting index
new_data.index = new_data.date
new_data.drop('date', axis=1, inplace=True)
# creating train and test sets
dataset = new_data.values

train = dataset[0:300, :]
valid = dataset[300:, :]

# converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i - 60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scaler.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)
rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
# for plotting
train = new_data[:300]
valid = new_data[300:]
valid['Predictions'] = closing_price
plt.plot(train['close'])
plt.plot(valid['close'], color='blue', label='True')
plt.plot(valid['Predictions'], color='red', label='Predict')
correct = 0
for i in range(len(closing_price)):
    if (valid['Predictions'][i] - valid['Predictions'][i-1]) * (valid['close'][i] - valid['close'][i-1]) > 0:
        correct += 1
print(correct/(len(closing_price)-1))
plt.legend()
plt.show()
