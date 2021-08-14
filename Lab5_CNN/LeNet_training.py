# coding:utf-8
import torch
import torch.nn as nn
import torch.optim as optime
from torchvision import datasets, transforms

from torch.autograd import Variable


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),  # 输出的维度是120
            # 没看懂 BatchNorm1d 是干啥的 但没了它训练效果很不好
            nn.BatchNorm1d(120),	nn.ReLU(),
            nn.Linear(120, 84), nn.BatchNorm1d(84), nn.ReLU(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


def Train():
    global device, batch_size, learnRate, Momentum, train_dataset, test_dataset, train_loader, test_loader
    global model, criterion, optimizer, num_classes
    for epoch in range(num_classes):
        print(epoch)

        # 通过enumerate的迭代，从训练集中取数据，i为索引，data为train_loader中的数据 # train_loader的长度为928，那么938*64=60032，正好是训练集数据的大小 # 这个循环一共会进行928次，到第900之后，就不会有数据打印出来了
        for i, data in enumerate(train_loader):
            # image为输入的数据，label为标签，即训练中给网络的基准答案
            image, labels = data
            image = image.cuda()
            labels = labels.cuda()
            # 将Tensor数据转换为Variable类型
            image, labels = Variable(image), Variable(labels)

            optimizer.zero_grad()  # 将梯度归零

            outputs = model(image)  # 将数据传入网络进行前向运算。这里只需要定义前向运算。

            loss = criterion(outputs, labels)  # 利用输出的预测值和标签之间的差值，得到损失函数的计算值
            loss.backward()  # loss反向传播
            optimizer.step()  # 使用优化器，对模型权重进行更新

            lossRate = loss.item()
            if i % 100 == 0:
                print('[{}-{} loss:{}]'.format(epoch + 1, i + 1, lossRate))
    torch.save(model, 'modelSave.pth')
    return model


def Test():
    correct = 0
    total = 0
# torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会 track 梯度。
    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            outputs = model(image)
            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()
    print('[Test Accuracy]: %.3f %%' % (100.0 * correct / total))


def init():
    global device, batch_size, learnRate, Momentum, train_dataset, train_loader
    global model, criterion, optimizer, num_classes, test_dataset, test_loader
    # 根据手册指示进行 device 设置：   device 为将要用于训练的硬件配置。
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda")

    # 每一次训练所取的样本数 学习率 动量
    # 主要需要调节的参数。多个教程中似乎这三个数字比较统一。
    batch_size = 64
    learnRate = 0.001
    Momentum = 0.9

    # 实例化一个LeNet网络，同时为其绑定设备。
    model = LeNet().to(device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 用的是SGD随机梯度下降算法， # 里面定义好网络的参数，学习率以及动量的大小 # 将网络的各种配置参数存入到优化器中，以便之后更新网络用
    optimizer = optime.SGD(model.parameters(), lr=learnRate, momentum=Momentum)

    num_classes = 10
    train_dataset = datasets.MNIST(root='./data', train=True,
                                   transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,  # 将读取的数据转变为TenSor
                                               batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='./data',  train=False,
                                  transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False)


init()
Train()
Test()
