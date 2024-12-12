import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


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


# img = train_data[50][0].numpy()
# label = train_data[50][1]
# plt.imshow(img[0, :])
# plt.show()


# ————————————定义网络———————————— #
# 5 * 5 * 4 -> 5 * 5 * 8 -> flatten -> fc
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 8, 5, padding=2)
        self.fc1 = nn.Linear((28 * 28) // (4 * 4) * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.bn1 = nn.BatchNorm2d(4, affine=True)
        self.bn2 = nn.BatchNorm2d(8, affine=True)

    # 前向传播
    def forward(self, x):
        # 5 * 5 * 4
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 5 * 5 * 8
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # flatten
        x = x.view(-1, (28 * 28) // (4 * 4) * 8)
        x = self.fc1(x)
        x = F.relu(x)

        # dropout层，只在训练时启用
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


# 读取已保存模型
MODEL_PATH = './MNIST_model_weights.pth'
net = model()
net.load_state_dict(torch.load(MODEL_PATH))
# 使用已保存模型时用不到下面这两个
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# ————————————训练网络———————————— #
# for epoch in range(20):
#     for i, data in enumerate(train_loader):
#         x, y = data
#         net.train()
#         pred = net(x)
#         loss = loss_fn(pred, y)
#
#         opt.zero_grad()
#         loss.backward()
#         opt.step()
#
#     print(epoch, loss)

# 计算准确率
rights = 0
length = 0
for i, data in enumerate(test_loader):
    x, y = data
    net.eval()
    pred = net(x)
    rights = rights + rightness(pred, y)[0]
    length = length + rightness(pred, y)[1]

print(f'{int(rights)} out of {int(length)}, accuracy is {float(rights / length) * 100:.2f}%')

# 满足要求时保存模型
# if float(rights / length) >= 0.96:
#     torch.save(net.state_dict(), "MNIST_model_weights.pth")
