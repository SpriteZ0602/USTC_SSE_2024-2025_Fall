import torch
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
x = torch.linspace(0, 10, steps=100, dtype=torch.float32)
# 添加噪声
rand = torch.randn(100) * 10
d = d * torch.ones_like(x)
y = a * x ** 3 + b * x ** 2 + c * x + d + rand
# ——————————划分训练测试集—————————— #
train_x = x[0: 90]
test_x = x[90: 100]
train_y = y[0: 90]
test_y = y[90: 100]

# 画出函数图
# plt.figure(figsize=(10, 8))
# plt.plot(train_x.data.numpy(), train_y.data.numpy(), 'o')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

# ——————————训练初始化—————————— #
# 初始化 a, b
a = torch.rand(1, requires_grad=True)
b = torch.rand(1, requires_grad=True)
c = torch.rand(1, requires_grad=True)
d = torch.rand(1, requires_grad=True)
# 学习率
rate = 1e-6
# 迭代数
epoch = 10000
# 记录loss
losses = []

# ——————————训练—————————— #
for i in range(epoch):
    pred = a.expand_as(train_x) * train_x ** 3 + b.expand_as(train_x) * train_x ** 2 + c.expand_as(train_x) * train_x + d.expand_as(train_x)
    # MSE
    loss = torch.mean((pred - train_y) ** 2)
    losses.append(loss.data)
    if i % 500 == 0:
        print("loss:", loss.data)
    # 反向传播、计算梯度
    loss.backward()
    a.data = a.data - a.grad.data * rate
    b.data = b.data - b.grad.data * rate
    c.data = c.data - c.grad.data * rate
    d.data = d.data - d.grad.data * rate
    a.grad.data.zero_()
    b.grad.data.zero_()
    c.grad.data.zero_()
    d.grad.data.zero_()

# ——————————结果—————————— #
print(f"{float(a)}x ^ 3 + {float(b)}x ^ 2 + {float(c)}x + {float(d)}")

# ——————————画出拟合图像—————————— #
plt.figure(figsize=(10, 8))
plt.plot(train_x.data.numpy(), train_y.data.numpy(), 'o')
plt.plot(train_x.data.numpy(), a.data.numpy() * train_x.data.numpy() ** 3 + b.data.numpy() * train_x.data.numpy() ** 2 + c.data.numpy() * train_x.data.numpy() + d.data.numpy())
plt.show()

# ——————————画出loss变化趋势—————————— #
plt.plot(losses)
plt.show()