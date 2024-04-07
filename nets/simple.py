from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FedAvgCNN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features,
                      32,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,
                      64,
                      kernel_size=5,
                      padding=0,
                      stride=1,
                      bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        self.fc1 = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(inplace=True)
        )
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.output(out)
        return out

class FedAvgCNN_BN(nn.Module):
    def __init__(self, in_features=3, num_classes=10, dim=1600):
        super().__init__()
        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(in_features,
                                    32,
                                    kernel_size=5,
                                    padding=0,
                                    stride=1,
                                    bias=True)),
                ('bn1', nn.BatchNorm2d(32)),
                ('relu1', nn.ReLU(inplace=True)),
                ('max1', nn.MaxPool2d(kernel_size=(2, 2))),
            ])
        )
        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv2', nn.Conv2d(32,
                        64,
                        kernel_size=5,
                        padding=0,
                        stride=1,
                        bias=True)),
                ('bn2', nn.BatchNorm2d(64)),
                ('relu2', nn.ReLU(inplace=True)),
                ('max', nn.MaxPool2d(kernel_size=(2, 2))),
            ])
        )
        self.fc1 = nn.Sequential(
            OrderedDict([
                ('linear', nn.Linear(dim, 512)),
                ('relu3', nn.ReLU(inplace=True)),
            ])
        )
        self.output = nn.Linear(512, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.output(out)
        return out

class DigitModel(nn.Module):
    """
    Model for benchmark experiment on Digits. 
    """
    def __init__(self, num_classes=10, **kwargs):
        super(DigitModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)
        self.output = nn.Linear(2048, num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = self.output(x)
        return x

class AlexNet(nn.Module):
    """
    used for DomainNet and Office-Caltech10
    """
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            OrderedDict([
                ('conv1', nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2)),
                ('bn1', nn.BatchNorm2d(64)),
                ('relu1', nn.ReLU(inplace=True)),
                ('maxpool1', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv2', nn.Conv2d(64, 192, kernel_size=5, padding=2)),
                ('bn2', nn.BatchNorm2d(192)),
                ('relu2', nn.ReLU(inplace=True)),
                ('maxpool2', nn.MaxPool2d(kernel_size=3, stride=2)),

                ('conv3', nn.Conv2d(192, 384, kernel_size=3, padding=1)),
                ('bn3', nn.BatchNorm2d(384)),
                ('relu3', nn.ReLU(inplace=True)),

                ('conv4', nn.Conv2d(384, 256, kernel_size=3, padding=1)),
                ('bn4', nn.BatchNorm2d(256)),
                ('relu4', nn.ReLU(inplace=True)),

                ('conv5', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
                ('bn5', nn.BatchNorm2d(256)),
                ('relu5', nn.ReLU(inplace=True)),
                ('maxpool5', nn.MaxPool2d(kernel_size=3, stride=2)),
            ])
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.output = nn.Linear(256 * 6 * 6, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.output(x)
        return x

class Softmax(nn.Module):
    def __init__(self, input_dim=1*28*28, num_classes=10):
        super(Softmax, self).__init__()
        self.output = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.output(x)
        output = F.log_softmax(x, dim=1)
        return output

class FemnistCNN(nn.Module):
    """
    Implements a model with two convolutional layers followed by pooling, and a final dense layer with 2048 units.
    Same architecture used for FEMNIST in "LEAF: A Benchmark for Federated Settings"__
    We use `zero`-padding instead of  `same`-padding used in
     https://github.com/TalwalkarLab/leaf/blob/master/models/femnist/cnn.py.
    """

    def __init__(self, num_classes):
        super(FemnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.fc1 = nn.Linear(64 * 4 * 4, 2048)
        self.output = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.output(x)
        return x

class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # batch*1*28*28（每次会送入batch个样本，输入通道数1（黑白图像），图像分辨率是28x28）
        # 下面的卷积层Conv2d的第一个参数指输入通道数，第二个参数指输出通道数，第三个参数指卷积核的大小
        self.conv1 = nn.Conv2d(1, 10, 5)  # 输入通道数1，输出通道数10，核的大小5
        self.conv2 = nn.Conv2d(10, 20, 3)  # 输入通道数10，输出通道数20，核的大小3
        # 下面的全连接层Linear的第一个参数指输入通道数，第二个参数指输出通道数
        self.fc1 = nn.Linear(20*10*10, 500)  # 输入通道数是2000，输出通道数是500
        self.output = nn.Linear(500, num_classes)  # 输入通道数是500，输出通道数是10，即10分类

    def forward(self, x):
        # 在本例中in_size=512，也就是BATCH_SIZE的值。输入的x可以看成是512*1*28*28的张量。
        in_size = x.size(0)
        # batch*1*28*28 -> batch*10*24*24（28x28的图像经过一次核为5x5的卷积，输出变为24x24）
        out = self.conv1(x)
        out = F.relu(out)  # batch*10*24*24（激活函数ReLU不改变形状））
        # batch*10*24*24 -> batch*10*12*12（2*2的池化层会减半）
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2(out)  # batch*10*12*12 -> batch*20*10*10（再卷积一次，核的大小是3）
        out = F.relu(out)  # batch*20*10*10
        # batch*20*10*10 -> batch*2000（out的第二维是-1，说明是自动推算，本例中第二维是20*10*10）
        out = out.view(in_size, -1)
        out = self.fc1(out)  # batch*2000 -> batch*500
        out = F.relu(out)  # batch*500
        out = self.output(out)  # batch*500 -> batch*10
        # out = F.log_softmax(out, dim=1)  # 计算log(softmax(x))
        return out
