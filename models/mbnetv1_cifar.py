"""
MobileNetV1 on CIFAR-10
"""
import torch
import torch.nn as nn
from methods import QConv2d, QLinear

class Net(nn.Module):
    """
    MobileNetV1 model for CIFAR-10
    """
    def __init__(self, alpha=1.0, num_classes=10, wbit=32, abit=32):
        super(Net, self).__init__()
        self.alpha = alpha   # width multiplier of the model

        def conv_bn(inp, oup, stride):
            layer = nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            return layer


        def conv_dw(inp, oup, stride):
            if wbit == 32:
                layer = nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            else:
                layer = nn.Sequential(
                    QConv2d(inp, inp, 3, stride, 1, groups=inp, bias=False, wbit=wbit, abit=abit),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    QConv2d(inp, oup, 1, 1, 0, bias=False, wbit=wbit, abit=abit),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            return layer

        self.model = nn.Sequential(
            conv_bn(3, int(32*self.alpha), 1, ), 
            conv_dw(int(32*self.alpha),  int(64*self.alpha), 1),
            conv_dw(int(64*self.alpha), int(128*self.alpha), 2),
            conv_dw(int(128*self.alpha), int(128*self.alpha), 1),
            conv_dw(int(128*self.alpha), int(256*self.alpha), 2),
            conv_dw(int(256*self.alpha), int(256*self.alpha), 1),
            conv_dw(int(256*self.alpha), int(512*self.alpha), 2),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(512*self.alpha), 1),
            conv_dw(int(512*self.alpha), int(1024*self.alpha), 2),
            conv_dw(int(1024*self.alpha), int(1024*self.alpha), 1),
        )
        self.pool = nn.AvgPool2d(2)
        self.fc = nn.Linear(int(1024*self.alpha), num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.pool(x)
        x = x.view(-1, int(1024*self.alpha))
        x = self.fc(x)
        return x

class mobilenetv1_Q:
    base=Net
    args = list()
    kwargs = {'alpha': 1.0}