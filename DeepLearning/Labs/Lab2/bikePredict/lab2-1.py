import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt


# ——————————数据集清洗—————————— #
# 离散数据处理
data = pd.read_csv("bikes.csv")
col_titles = ["season", "weathersit", "mnth", "hr", "weekday"]
for i in col_titles:
    dummies = pd.get_dummies(data[i], prefix=1)
    data = pd.concat([data, dummies], axis=1)
col_titles_to_drop = ["instant", "dteday", "season", "weathersit", "weekday", "mnth", "workingday", "hr"]
data = data.drop(col_titles_to_drop, axis=1)

# 连续数据处理（归一化）
col_titles = ["cnt", "temp", "hum", "windspeed"]
for i in col_titles:
    mean, std = data[i].mean(), data[i].std()
    if i == "cnt":
        mean_cnt, std_cnt = mean, std
    data[i] = (data[i] - mean) / std

# 分割训练测试集
test_data = data[-30 * 24:]
train_data = data[:-30 * 24]
# 提取label
X = train_data.drop(["cnt"], axis=1)
X = X.values.astype(float)
Y = train_data["cnt"]
Y = Y.values.astype(float)
Y = np.reshape(Y, [len(Y), 1])

# ——————————神经网络搭建—————————— #
# 超参数定义
input_size = X.shape[1]
batch_size = 128
epoch = 1000
# 神经网络定义
NEU = torch.nn.Sequential(
    torch.nn.Linear(input_size, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128,256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1)
)
loss_fn = torch.nn.MSELoss()
opt = optim.SGD(NEU.parameters(), lr=0.01)

# ——————————训练—————————— #
# 记录loss
losses = []
for i in range(epoch):
    batch_loss = []
    for start in range(0, len(X), batch_size):
        if start + batch_size < len(X):
            end = start + batch_size
        else:
            end = len(X)
        x = torch.FloatTensor(X[start: end])
        y = torch.FloatTensor(Y[start: end])
        pred = NEU(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        batch_loss.append(loss.data.numpy())
    if i % 100 == 0:
        losses.append(np.mean(batch_loss))
        print(f'epoch:{i}, loss:{np.mean(batch_loss)}')

# 绘制loss变化趋势
plt.plot(np.arange(len(losses)) * 100, losses)
plt.xlabel("epoch")
plt.ylabel("MSE")
plt.show()

# ——————————验证—————————— #
# 预测训练集
# X = train_data.drop(['cnt'], axis=1)
# Y = train_data['cnt']
# 预测测试集
X = test_data.drop(['cnt'], axis=1)
Y = test_data['cnt']
Y = Y.values.reshape([len(Y), 1])
X = X.values.astype(float)
X = torch.FloatTensor(X)
Y = torch.FloatTensor(Y)
pred = NEU(X)

# 反归一化
Y = Y.data.numpy() * std_cnt + mean_cnt
pred = pred.data.numpy() * std_cnt + mean_cnt

# 绘制预测图象
plt.figure(figsize=(10, 7))
xplot, = plt.plot(np.arange(X.size(0)), Y)
yplot, = plt.plot(np.arange(X.size(0)), pred, ':')
plt.show()