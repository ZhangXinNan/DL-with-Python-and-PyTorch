
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from collections import Counter

#定义一些超参数 
BATCHSIZE = 100
DOWNLOAD_MNIST = False
EPOCHES = 100
LR = 0.001


#定义相关模型结构，这三个网络结构比较接近
#导入数据，这里数据已下载本地，故设download=False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
from net_cnn import CNNNet
from net1 import Net
from net2 import LeNet
# from net_vgg import VGG


net1 = CNNNet()
net2 = Net()
net3 = LeNet()
# net4 = VGG('VGG16')

#把3个网络模型放在一个列表里
mlps=[net1.to(device), net2.to(device), net3.to(device)]

optimizer=torch.optim.Adam([{"params":mlp.parameters()} for mlp in mlps],lr=LR)
  
loss_function=nn.CrossEntropyLoss()
 
for ep in range(EPOCHES):
    for img,label in trainloader:
        img,label=img.to(device),label.to(device)
        optimizer.zero_grad()#10个网络清除梯度
        for mlp in mlps:
            mlp.train()
            out=mlp(img)
            loss=loss_function(out,label)
            loss.backward()#网络们获得梯度
        optimizer.step()
 
    pre=[]
    vote_correct=0
    mlps_correct=[0 for i in range(len(mlps))]
    for img,label in testloader:
        img,label=img.to(device),label.to(device)
        for i, mlp in  enumerate( mlps):
            mlp.eval()
            out=mlp(img)
 
            _,prediction=torch.max(out,1) #按行取最大值
            pre_num=prediction.cpu().numpy()
            mlps_correct[i]+=(pre_num==label.cpu().numpy()).sum()
 
            pre.append(pre_num)
        arr=np.array(pre)
        pre.clear()
        result=[Counter(arr[:,i]).most_common(1)[0][0] for i in range(BATCHSIZE)]
        vote_correct+=(result == label.cpu().numpy()).sum()
    print("epoch:" + str(ep)+"集成模型的正确率"+str(vote_correct/len(testloader)))
 
    for idx, coreect in enumerate( mlps_correct):
        print("模型"+str(idx)+"的正确率为："+str(coreect/len(testloader)))



