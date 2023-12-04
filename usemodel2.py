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
model = load_model('modelalldata.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

df = pd.read_csv('stock_prices.csv')
data = df.filter(['open', 'close', 'high', 'low', 'volume', 'amount'])
df['date'] = pd.to_datetime(df.date, format='%Y%m%d')
df.index = df['date']
print(data.head())
dataset = data.values  # 转化为 Numpy 数组
training_data_len = int(np.ceil(0.8 * len(dataset)))  # 取数据集前80%作为训练数据
scaler = MinMaxScaler(feature_range=(0, 1))  # 将数据规范化到 0 到 1 的范围
scaled_data = scaler.fit_transform(dataset)
print(dataset.shape)
dataset1 = dataset
test_data = scaled_data[training_data_len - 60:, :]
x_test = []
y_test = dataset[training_data_len:, 1]  # 这里假设我们只预测开盘价，所以取第 0 列

for i in range(60, len(test_data)):
    x_test.append(test_data[i - 60:i, :])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))
prelist = np.array(x_test)
predictions = model.predict(x_test)
for i in range(predictions.shape[0]):
    dataset1[i][1] = predictions[i]
prelist1 = scaler.inverse_transform(dataset1)
for i in range(predictions.shape[0]):
    predictions[i] = prelist1[i][1]
# 可视化结果
correct = 0
for i in range(1, predictions.shape[0]):
    if (predictions[i] - predictions[i-1]) * (y_test[i] - y_test[i-1]) > 0:
        correct += 1
print(correct/(predictions.shape[0]-1))
t = df['date'].iloc[training_data_len:]
plt.plot(t, predictions, color='red', label='Predict')
plt.plot(t, y_test, color='blue', label='True')
plt.legend()
plt.show()
