import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.conv2d_drop = nn.Dropout()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv1(x))))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv2d_drop(self.conv2(x))))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SoftmaxClassifier(nn.Module):
    def __init__(self):
        super(SoftmaxClassifier, self).__init__()
        self.fc = nn.Linear(28 * 28 * 3, 14, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_().mul_(0.005)

    def forward(self, x):
        x = x.view(-1, 28 * 28 * 3)
        x = self.fc(x)
        return x
