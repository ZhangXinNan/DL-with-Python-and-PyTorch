import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import load_data_cifar, classes
import argparse

device = None
# device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
# print(device)


def print_accuracy_of_classes(net, testloader):
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
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


def val(net, testloader, criterion):
    correct = 0
    loss = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels)
            # print(images.size(), labels.size(), predicted.size())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print('Accuracy of the network on the 10000 test images : %d %%' %(100 * correct/total))
    return correct / total, loss / total


def train(net, trainloader, testloader, num_epoch, optimizer, criterion):
    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])

    for epoch in range(num_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            batch_size = inputs.size()[0]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            running_acc += (predicted == labels).sum().item()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                acc, val_loss = val(net, testloader, criterion)
                str_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                train_acc = running_acc / (2000 * batch_size)
                train_loss = running_loss / 2000
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                print('%s [%d, %5d] train acc: %.3f loss: %.3f ; val acc: %.3f loss: %.3f  lr: %.3f' % (str_time, epoch + 1, i + 1, train_acc, train_loss, acc, val_loss, lr))
                # print('\t', optimizer.state_dict()['param_groups'])
                running_loss = 0.0
                running_acc = 0
    print('Finished Training')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cnn', help='cnn, gap, vgg')
    parser.add_argument('--num_epoch', default=10, type=int)
    parser.add_argument('--gpu_id', default=0, type=int)
    return parser.parse_args()


def main(args):
    global device
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{}".format(args.gpu_id))
    else:
        device = torch.device("cpu")

    if args.model == 'cnn':
        from net_cnn import CNNNet as Net
        net = Net()
    elif args.model == 'gap':
        from net_gap import Net
        net = Net()
    elif args.model == 'vgg':
        from net_vgg import VGG
        net = VGG()

    # device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    # print(device)
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    trainloader, testloader = load_data_cifar()
    weights_file = './cifar_net_{}.pth'.format(args.model)
    if not os.path.isfile(weights_file):
        train(net, trainloader, testloader, args.num_epoch, optimizer, criterion)
        # PATH = './cifar_net.pth'
        torch.save(net.state_dict(), weights_file)
    else:
        net.load_state_dict(torch.load(weights_file))
        net.to(device)
        val_acc, val_loss = val(net, testloader, criterion)
        print('{} : val_acc : {}, val_loss : {}'.format(args.model, val_acc, val_loss))
    # 打印每个类别的准确率
    print_accuracy_of_classes(net, testloader)


if __name__ == '__main__':
    main(get_args())
