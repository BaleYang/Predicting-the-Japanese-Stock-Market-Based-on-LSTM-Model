import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

# 读取数据集
data = pd.read_csv("datas.csv")


# 读取数据集
X = data.iloc[:, :-1]
y = data.iloc[:, -1]


# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# 选择k个最好的特征
selector = SelectKBest(f_regression, k=20)
X_new = selector.fit_transform(X_scaled, y)

feature_names = ['feature_' + str(i) for i in range(len(X_new[0]))]

# 将X和y合并成一个DataFrame对象并指定列名
data_processed = pd.DataFrame(data=X_new, columns=feature_names)
data_processed['target'] = y
data_processed.to_csv('processed_datas.csv', header=False, index=False)
print('数据预处理完成！')
