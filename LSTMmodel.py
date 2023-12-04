import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
print()

df = pd.read_csv("/all_stock_prices.csv")  # 假设数据集文件名为 stock_data.csv
data = df.filter(['open', 'close', 'high', 'low', 'volume', 'amount', 'com_code'])
dataset = data.values
x_train = []
y_train = []
grouped_data = df.groupby('com_code')
scaler = MinMaxScaler(feature_range=(0, 1))
for group_name, group_data in grouped_data:
    group_data_trainlen = int(0.8 * len(group_data))
    group_data_train = group_data.values
    scaled_data = scaler.fit_transform(group_data_train)
    for i in range(60, group_data_trainlen):
        x_train.append(scaled_data[i - 60:i, 1:7])
        y_train.append(scaled_data[i, 1])  # 这里我们只预测收盘价，所以取第 1 列
x_train, y_train = np.array(x_train), np.array(y_train)

model = Sequential()

model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 6)))
model.add(Dropout(0.2))

model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=128))
model.add(Dropout(0.2))

model.add(Dense(units=1))  # 输出层

model.compile(optimizer='adam', loss='mean_squared_error')

# 训练模型
model.fit(x_train, y_train, epochs=200, batch_size=1)
model.save('/model.h5')
