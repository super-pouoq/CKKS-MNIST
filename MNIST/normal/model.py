import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.AvgPool2d(2) # 替换 MaxPool
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.AvgPool2d(2) # 替换 MaxPool
        
        self.flatten = nn.Flatten()
        
        # 全连接层
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # 替换 ReLU 为 x 的平方
        x = self.pool1((self.conv1(x)) ** 2)
        x = self.pool2((self.conv2(x)) ** 2)
        x = self.flatten(x)
        x = self.fc1(x) ** 2
        x = self.fc2(x)
        return x