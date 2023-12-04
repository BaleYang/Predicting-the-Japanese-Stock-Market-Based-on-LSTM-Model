import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# 读取数据集
df = pd.read_csv("all_stock_prices.csv")  # 假设数据集文件名为 stock_data.csv
data = df.filter(['open', 'close', 'high', 'low', 'volume', 'amount', 'com_code'])

print(data.head())
dataset = data.values
x_train = []
y_train = []
grouped_data = df.groupby('com_code')
print(grouped_data)
scaler = MinMaxScaler(feature_range=(0, 1))
for group_name, group_data in grouped_data:
    group_data_trainlen = int(0.8 * len(group_data))
    group_data_train = group_data.values
    scaled_data = scaler.fit_transform(group_data_train)
    for i in range(60, group_data_trainlen):
        x_train.append(scaled_data[i - 60:i, 1:7])
        y_train.append(scaled_data[i, 2])  # 这里我们只预测收盘价，所以取第 1 列
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train)
print(x_train.shape)
print(y_train)
print(y_train.shape)
