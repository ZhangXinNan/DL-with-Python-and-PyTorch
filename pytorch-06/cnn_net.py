
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 36, 3)
        self.fc1 = nn.Linear(1296, 128) # 1296 = 6 * 6 * 36
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 36 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


def init_weigths(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight)
            nn.init.xavier_normal_(m.weight)
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight)


def main():
    net = Net()
    print("查看网络结构：")
    print(net)

    print("查看网络前几层：")
    print(nn.Sequential(*list(net.children()))[:4])


if __name__ == '__main__':
    main()
