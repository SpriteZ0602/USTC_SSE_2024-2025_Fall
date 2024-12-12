import torch
from torchvision import transforms, datasets
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


train_data = datasets.CIFAR10(root='./',
                              train=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
                                  # 随机裁剪
                                  transforms.RandomHorizontalFlip(p=0.2)
                            ]),
                            download=True
                            )
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)

test_data = datasets.CIFAR10(root='./',
                             train=False,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
                             ]))
test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True)


class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv11 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv31 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64, affine=True)
        self.bn2 = nn.BatchNorm2d(128, affine=True)
        self.bn3 = nn.BatchNorm2d(256, affine=True)
        self._initialize_fc()
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 3 * 3 * 64
        x = self.conv11(x)
        x = self.conv12(x)
        # bn层
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # 3 * 3 * 128
        x = self.conv21(x)
        x = self.conv22(x)
        # bn层
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # 3 * 3 * 256
        x = self.conv31(x)
        x = self.conv32(x)
        # bn层
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # flatten
        x = x.view(-1, self.flattened_size)
        x = self.fc1(x)
        x = F.relu(x)

        # dropout层
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

    def _initialize_fc(self):
        # 创建一个虚拟输入计算展平后的特征图尺寸，计算Flatten层拉伸size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 32, 32)
            x = self.conv11(dummy_input)
            x = self.conv12(x)
            x = self.pool(x)
            x = self.conv21(x)
            x = self.conv22(x)
            x = self.pool(x)
            x = self.conv31(x)
            x = self.conv32(x)
            x = self.pool(x)
            self.flattened_size = x.numel()


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


# 读取已保存模型
MODEL_PATH = './CIFAR10_model_weights.pth'
net = model()
net.load_state_dict(torch.load(MODEL_PATH))
# 使用已保存模型时用不到下面这两个
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

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
# if float(rights / length) >= 0.8:
#     torch.save(net.state_dict(), "CIFAR10_model_weights.pth")
