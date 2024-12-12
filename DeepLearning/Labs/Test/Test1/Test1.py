import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

data = pd.read_csv('../test.csv')

# 预处理
X = data.iloc[:, 1:-1].values  # 不提取第一列
y = data.iloc[:, -1].values  # label

# 8：2划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)


class model(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(model, self).__init__()
        self.fc1_SA465 = nn.Linear(input_size, hidden_size1)
        self.relu1_SA465 = nn.ReLU()
        self.fc2_SA465 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2_SA465 = nn.ReLU()
        self.fc3_SA465 = nn.Linear(hidden_size2, num_classes)

    def forward(self, x):
        x = self.fc1_SA465(x)
        x = self.relu1_SA465(x)
        x = self.fc2_SA465(x)
        x = self.relu2_SA465(x)
        x = self.fc3_SA465(x)
        return x


# 参数
input_size_SA465 = X_train.shape[1]
hidden_size1_SA465 = 64
hidden_size2_SA465 = 32
num_classes_SA465 = len(np.unique(y))

net_SA465 = model(input_size_SA465, hidden_size1_SA465, hidden_size2_SA465, num_classes_SA465)

# 交叉熵损失函数 Adam优化器
criterion_SA465 = nn.CrossEntropyLoss()
optimizer_SA465 = optim.Adam(net_SA465.parameters(), lr=0.001)

# 训练
num_epochs_SA465 = 50
batch_size_SA465 = 32

for epoch in range(num_epochs_SA465):
    net_SA465.train()
    for i in range(0, len(X_train), batch_size_SA465):
        x_batch = X_train[i:i + batch_size_SA465]
        y_batch = y_train[i:i + batch_size_SA465]

        # 前向传播
        outputs = net_SA465(x_batch)
        loss_SA465 = criterion_SA465(outputs, y_batch)

        # 反向传播
        optimizer_SA465.zero_grad()
        loss_SA465.backward()
        optimizer_SA465.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch: {epoch + 1}, Loss: {loss_SA465.data:.4f}')

# 测试
net_SA465.eval()
with torch.no_grad():
    y_pred = net_SA465(X_test)
    y_pred_classes = torch.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test, y_pred_classes)
    print(f'准确率: {acc * 100}%')

if acc >= 0.96:
    torch.save(net_SA465.state_dict(), "Test1_model.pth")
