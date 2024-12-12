import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset


# 生成序列 hn+1 = 3 * hn + 2
def getSeq(start, n):
    x = [3 * x + 2 for x in range(start, start + n)]
    return x


data = []

# 生成长度为5的序列
for i in range(100):
    rnd = np.random.randint(0, 25)
    data.append(getSeq(rnd, 6))

data = np.array(data)
data = torch.from_numpy(data)

# 制作训练集、验证集
target = data[:, -1:].type(torch.FloatTensor)
data = data[:, :-1].type(torch.FloatTensor)

train_x = data[: 90]
train_y = target[: 90]
test_x = data[90:]
test_y = target[90:]

train_dataset = TensorDataset(train_x, train_y)
test_dataset = TensorDataset(test_x, test_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=5, shuffle=True)


# 定义模型
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.LSTM = nn.LSTM(1, 10, batch_first=True)
        self.FC = nn.Linear(10, 1)

    def forward(self, x, hidden):
        output, hidden = self.LSTM(x, hidden)
        output = output[:, -1, :]
        output = self.FC(output)
        return output, hidden


net = model()
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(net.parameters(), lr=0.001)

# 定义h0、c0
h0 = torch.zeros(1, 5, 10)
c0 = torch.zeros(1, 5, 10)

for epoch in range(1000):
    for i, data in enumerate(train_loader):
        x, y = data
        x = x.view(-1, 5, 1)

        hidden = (h0, c0)
        pred, hidden = net(x, hidden)
        loss = loss_fn(pred, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")

for i, data in enumerate(test_loader):
    x, y = data
    x = x.view(x.size(0), x.size(1), 1)

    h0 = torch.zeros(1, x.size(0), 10)
    c0 = torch.zeros(1, x.size(0), 10)
    hidden = (h0, c0)

    pred, hidden = net(x, hidden)
    print(f'Expected: {y.view(1, -1).data}')
    print(f'Predicted: {pred.view(1, -1).data}')


