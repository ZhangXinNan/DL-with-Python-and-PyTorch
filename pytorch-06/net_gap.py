
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 5)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.aap = nn.AdaptiveAvgPool2d(1)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(36, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.aap(x)
        # x = x.view(-1, 16 * 5 * 5)
        x = x.view(x.shape[0], -1)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def main():
    net = Net()
    print("查看网络结构：")
    print(net)

    print("查看网络前几层：")
    print(nn.Sequential(*list(net.children()))[:4])


if __name__ == '__main__':
    main()


