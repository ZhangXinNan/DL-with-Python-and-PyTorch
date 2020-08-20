import time
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_data_cifar, classes
import argparse


device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(device)


def val(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(images.size(), labels.size(), predicted.size())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the 10000 test images : %d %%' %(100 * correct/total))
    return 100 * correct/total


def train(net, trainloader, testloader, num_epoch):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                acc = val(net, testloader)
                str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print('%s [%d, %5d] loss: %.3f accuracy: %.3f' % (str_time, epoch + 1, i + 1, running_loss / 2000, acc))
                running_loss = 0.0
    print('Finished Training')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='lenet5', help='cnn, gap, vgg')
    parser.add_argument('--num_epoch', default=10, type=int)
    return parser.parse_args()


def main(args):
    if args.model == 'lenet5':
        from net import Net
        net = Net()
    elif args.model == 'cnn':
        from cnn_net import Net
        net = Net()
    elif args.model == 'gap':
        from net_gap import Net
        net = Net()
    elif args.model == 'vgg':
        from vgg import VGG
        net = VGG()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # print(device)
    net.to(device)

    trainloader, testloader = load_data_cifar()
    train(net, trainloader, testloader, args.num_epoch)

    # PATH = './cifar_net.pth'
    torch.save(net.state_dict(), './cifar_net_{}.pth'.format(args.model))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    main(get_args())
