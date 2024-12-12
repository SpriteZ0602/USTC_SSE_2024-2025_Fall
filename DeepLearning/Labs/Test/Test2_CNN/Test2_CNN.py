import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F


# ————————————读取MNIST数据集———————————— #
train_data = datasets.MNIST(root='./',
                            train=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize([0.1307, ], [0.3081, ])
                            ]),
                            download=True
                            )
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.MNIST(root='./',
                           train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize([0.1307, ], [0.3081])
                           ]))
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)


# ————————————定义网络———————————— #
# 5 * 5 * 4 -> 5 * 5 * 8 -> flatten -> fc
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_SA465 = nn.Conv2d(1, 4, 5, padding=2)
        self.pool_SA465 = nn.MaxPool2d(2, 2)
        self.conv2_SA465 = nn.Conv2d(4, 8, 5, padding=2)
        self.fc1_SA465 = nn.Linear((28 * 28) // (4 * 4) * 8, 512)
        self.fc2_SA465 = nn.Linear(512, 10)
        self.bn1_SA465 = nn.BatchNorm2d(4, affine=True)
        self.bn2_SA465 = nn.BatchNorm2d(8, affine=True)

    # 前向传播
    def forward(self, x):
        # 5 * 5 * 4
        x = self.conv1_SA465(x)
        x = self.bn1_SA465(x)
        x = F.relu(x)
        x = self.pool_SA465(x)

        # 5 * 5 * 8
        x = self.conv2_SA465(x)
        x = self.bn2_SA465(x)
        x = F.relu(x)
        x = self.pool_SA465(x)

        # flatten
        x = x.view(-1, (28 * 28) // (4 * 4) * 8)
        x = self.fc1_SA465(x)
        x = F.relu(x)

        # dropout层，只在训练时启用
        x = F.dropout(x, training=self.training)
        x = self.fc2_SA465(x)
        return x


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_SA465 = model().to(device)
loss_fn_SA465 = nn.CrossEntropyLoss()
opt_SA465 = torch.optim.SGD(net_SA465.parameters(), lr=0.001, momentum=0.9)

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
    torch.save(net_SA465.state_dict(), "Test2_CNN.pth")
