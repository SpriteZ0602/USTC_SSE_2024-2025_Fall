import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
import matplotlib.pyplot as plt

# ——————————初始化函数—————————— #
# 设置 y = ax^3 + bx^2 + cx + d 中 a, b, c, d 的值
a = input("input a:")
b = input("input b:")
c = input("input c:")
d = input("input d:")
a = int(a)
b = int(b)
c = int(c)
d = int(d)
x = torch.linspace(0, 10, 100, dtype=torch.float32)
# 添加噪声
rand = torch.randn(100) * 5
y = a * x ** 3 + b * x ** 2 + c * x + d + rand
# ——————————划分训练测试集—————————— #
train_x = x[0: 80]
test_x = x[80: 100]
train_y = y[0: 80]
test_y = y[80: 100]


# ——————————定义模型—————————— #
class model(nn.Module):
    # 定义神经网络，四层线性层
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 5)
        self.hidden2 = nn.Linear(5, 10)
        self.hidden3 = nn.Linear(10, 10)
        self.hidden4 = nn.Linear(10, 5)
        self.out = nn.Linear(5, 1)

    # 定义前向传播
    def forward(self, x):
        x = self.hidden1(x)
        x = func.relu(x)
        x = self.hidden2(x)
        x = func.relu(x)
        x = self.hidden3(x)
        x = func.relu(x)
        x = self.hidden4(x)
        x = func.relu(x)
        x = self.out(x)
        return x


# ——————————训练—————————— #
# 迭代数
epoch = 10000
net = model()
# 损失函数MSE
loss_fn = torch.nn.MSELoss()
# 记录loss
losses = []
opt = torch.optim.SGD(net.parameters(), lr=1e-7, momentum=0.9)
for i in range(epoch):
    pred = net(train_x.view(-1, 1))
    loss = loss_fn(pred, train_y.view(-1, 1))
    losses.append(loss.data)
    if i % 500 == 0:
        print("loss: ", loss.data)

    opt.zero_grad()
    loss.backward()
    opt.step()

# ——————————结果—————————— #
# 画出训练集拟合结果
plt.figure(figsize=(10, 8))
plt.plot(train_x.data.numpy(), train_y.data.numpy(), 'o')
plt.plot(train_x.data.numpy(), net(train_x.view(-1, 1)).data.numpy())
plt.show()

# 画出测试集拟合结果
pred = net(test_x.view(-1, 1))
plt.figure(figsize=(5, 5))
plt.plot(test_x.data.numpy(), test_y.data.numpy(), 'o')
plt.plot(test_x.data.numpy(), pred.data.numpy())
plt.show()

# 画出loss
plt.plot(losses)
plt.show()