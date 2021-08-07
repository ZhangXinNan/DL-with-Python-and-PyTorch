import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.aap=nn.AdaptiveAvgPool2d(1)
        #self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #print(x.shape)
        #x = x.view(-1, 16 * 5 * 5)
        x = self.aap(x)
        #print(x.shape)
        #x = F.relu(self.fc2(x))
        x = x.view(x.shape[0], -1)
        #print(x.shape)
        x = self.fc3(x)
        return x