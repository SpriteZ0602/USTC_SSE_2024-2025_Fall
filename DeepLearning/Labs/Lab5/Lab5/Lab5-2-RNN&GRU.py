import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# 数据生成
data = []
start = 0
for i in range(200):
    x = [np.sin(x / 10) for x in range(start, start + 11)]
    data.append(x)
    start += 1

data = np.array(data, dtype=np.float32)
data = torch.tensor(data)

target = data[:, -1:]  # 最后一列为目标
data = data[:, :-1]    # 前10列为输入

# 数据归一化
data_max, data_min = data.max(), data.min()
data = (data - data_min) / (data_max - data_min)
target = (target - data_min) / (data_max - data_min)

# 数据集划分
train_x, train_y = data[:150], target[:150]
test_x, test_y = data[150:], target[150:]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)


# 定义模型，GRU与RNN层定义方法类似
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.GRU = nn.GRU(input_size=1, hidden_size=20, num_layers=2, batch_first=True, dropout=0.2)
        self.RNN = nn.RNN(input_size=1, hidden_size=20, num_layers=2, batch_first=True, dropout=0.2)
        self.FC = nn.Linear(20, 1)

    def forward(self, x, hidden):
        # output, hidden = self.GRU(x, hidden)
        output, hidden = self.RNN(x, hidden)
        output = self.FC(output[:, -1, :])  # 取最后一个时间步
        return output, hidden


losses = []
net = Model()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=0.01)


def init_hidden(batch_size):
    return torch.zeros(2, batch_size, 20)


# 训练
for epoch in range(300):
    for x, y in train_loader:
        x = x.view(-1, 10, 1)
        hidden = init_hidden(x.size(0))

        pred, _ = net(x, hidden)
        loss = loss_fn(pred, y)
        losses.append(loss.data)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")


# 预测
net.eval()
preds = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.view(-1, 10, 1)
        hidden = init_hidden(x.size(0))
        pred, _ = net(x, hidden)
        preds.append(pred.numpy())

preds = np.concatenate(preds, axis=0)
data_max = data_max.item()
data_min = data_min.item()


# 反归一化
preds = preds * (data_max - data_min) + data_min
train_y = train_y * (data_max - data_min) + data_min
test_y = test_y * (data_max - data_min) + data_min


plt.figure(figsize=(10, 6))
plt.scatter(range(len(train_y)), train_y, marker='o')
plt.scatter(range(len(train_y), len(train_y) + len(test_y)), preds, marker='s')
# plt.title('GRU')
plt.title('RNN')
plt.show()

plt.plot(losses)
# plt.title('GRU Loss')
plt.title('RNN Loss')
plt.show()
