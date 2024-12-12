import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import DataLoader

# ————————————读取MNIST数据集———————————— #
train_data = datasets.MNIST(
    root='./',
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307, ], [0.3081, ])
    ]),
    download=True
)
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(
    root='./',
    train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.1307, ], [0.3081, ])
    ])
)
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)


# ————————————定义网络———————————— #
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm_SA465 = nn.LSTM(input_size=28, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)
        self.fc_SA465 = nn.Linear(128, 10)

    def forward(self, x):
        # 将图像数据转换为序列数据
        x = x.squeeze(1)
        out, _ = self.lstm_SA465(x)
        out = out[:, -1, :]  # 取最后一个时间步的输出
        out = self.fc_SA465(out)
        return out


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_SA465 = model().to(device)
loss_fn_SA465 = nn.CrossEntropyLoss()
opt_SA465 = torch.optim.Adam(net_SA465.parameters(), lr=0.001)

# ————————————训练网络———————————— #
for epoch in range(20):
    for i, data in enumerate(train_loader):
        x, y = data
        x, y = x.to(device), y.to(device)

        net_SA465.train()
        pred = net_SA465(x)
        loss = loss_fn_SA465(pred, y)

        opt_SA465.zero_grad()
        loss.backward()
        opt_SA465.step()

    print(f'Epoch: {epoch + 1} Loss: {loss.data:.4f}')

# 计算准确率
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    x, y = x.to(device), y.to(device)
    net_SA465.eval()
    pred = net_SA465(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]

print(f'{int(rights)} out of {int(length)}, accuracy is {float(rights / length) * 100:.2f}%')

if float(rights / length) >= 0.96:
    torch.save(net_SA465.state_dict(), "Test2_LSTM.pth")
