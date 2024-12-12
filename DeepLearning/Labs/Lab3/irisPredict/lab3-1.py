import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# ————————————聚类算法生成数据———————————— #
data, target = make_blobs(n_samples=500, n_features=2, centers=[[2, 1], [3, 3], [4, 2]],cluster_std=[0.3, 0.2, 0.2], random_state=8)
# 画图
plt.scatter(data[:, 0], data[:, 1], c=target, marker='o')
plt.show()

# ————————————数据准备———————————— #
data = torch.from_numpy(data)
data = data.type(torch.FloatTensor)
target = torch.from_numpy(target)
target = target.type(torch.LongTensor)
# 分割训练测试集
train_x = data[:400]
train_y = target[:400]
test_x = data[400:]
test_y = target[400:]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=True)


# ————————————定义网络———————————— #
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(2, 5)
        self.out = nn.Linear(5, 3)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.out(x)
        return x


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


net = model()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.01)
# 记录loss
losses = []

# ————————————训练网络———————————— #
for epoch in range(1000):
    for i, data in enumerate(train_loader):
        x, y = data
        pred = net(x)
        loss = loss_fn(pred, y)
        losses.append(loss.data)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 100 == 0:
        print(loss)

# 计算准确率
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]
    # print(y)
    # print(torch.max(pred.data, 1)[1], '\n')

print(f'{int(rights)} out of {int(length)}, accuracy is {float(rights / length) * 100}%')

plt.plot(losses)
plt.show()