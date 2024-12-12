import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# ————————————定义网络———————————— #
class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(4, 5)
        self.hidden2 = nn.Linear(5, 10)
        self.hidden3 = nn.Linear(10, 5)
        self.out = nn.Linear(5, 3)

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = self.hidden3(x)
        x = F.relu(x)
        x = self.out(x)
        return x


def rightness(pred, labels):
    pred = torch.max(pred.data, 1)[1]
    rights = pred.eq(labels.data.view_as(pred)).sum()
    return rights, len(labels)


if __name__ == '__main__':
    lr = 0.01
    epoches = 500
    losses = []
    MODEL_PATH = './iris_model_state_dict.pth'

    # ————————————数据集清洗———————————— #
    data = pd.read_csv('iris.csv')
    for i in range(len(data)):
        if data.loc[i, 'Species'] == 'setosa':
            data.loc[i, 'Species'] = 0
        if data.loc[i, 'Species'] == 'versicolor':
            data.loc[i, 'Species'] = 1
        if data.loc[i, 'Species'] == 'virginica':
            data.loc[i, 'Species'] = 2

    data = data.drop('Unnamed: 0', axis=1)
    data = shuffle(data)
    data.index = range(len(data))

    col_titles = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
    for i in col_titles:
        mean, std = data[i].mean(), data[i].std()
        data[i] = (data[i] - mean) / std
    # 分割训练测试集
    train_data = data[: -32]
    train_x = train_data.drop(['Species'], axis=1)
    train_y = train_data['Species'].values.astype(int)
    train_x = train_x.values.astype(float)
    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    test_data = data[-32:]
    test_x = train_data.drop(['Species'], axis=1)
    test_y = train_data['Species'].values.astype(int)
    test_x = test_x.values.astype(float)
    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)

    train_dataset = TensorDataset(train_x, train_y)
    test_dataset = TensorDataset(test_x, test_y)
    train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=True)

    # ————————————训练网络———————————— #
    # net = model()
    # loss_fn = nn.CrossEntropyLoss()
    # opt = torch.optim.SGD(net.parameters(), lr=lr)
    # for epoch in range(epoches):
    #     for i, data in enumerate(train_loader):
    #         x, y = data
    #         pred = net(x)
    #         loss = loss_fn(pred, y)
    #         losses.append(loss.data)
    #
    #         opt.zero_grad()
    #         loss.backward()
    #         opt.step()
    #     if epoch % 10 == 0:
    #         print(f'{epoch}: {loss}')

    # 加载模型
    net = model()
    net.load_state_dict(torch.load(MODEL_PATH))

    # 正确率计算
    rights = 0
    length = 0
    for i, data in enumerate(test_loader):
        x, y = data
        pred = net(x)
        rights = rights + rightness(pred, y)[0]
        length = length + rightness(pred, y)[1]
        # print(y)
        # print(torch.max(pred.data, 1)[1], '\n')

    print(f'{int(rights)} out of {length}, accuracy is {float(rights / length) * 100}%')

    # 绘制loss趋势
    # plt.plot(losses)
    # plt.show()

    # 保存模型
    # if float(rights / length) >= 0.99:
    # torch.save(net, MODEL_PATH)
    # print('Model Saved!')
