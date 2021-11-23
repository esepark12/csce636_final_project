
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_features, out_features, stride, t=6, class_num=100):
        super().__init__()

        self.resblock = nn.Sequential( # residual block
            nn.Conv2d(in_features, t * in_features, 1),
            nn.BatchNorm2d(t * in_features),
            nn.ReLU6(inplace=True),

            nn.Conv2d(t * in_features, t * in_features, 3, stride=stride, padding=1, groups=t * in_features),
            nn.BatchNorm2d(t * in_features),
            nn.ReLU6(inplace=True),

            nn.Conv2d(t * in_features, out_features, 1),
            nn.BatchNorm2d(out_features)
        )
        self.out_channels = out_features
        self.in_channels = in_features
        self.stride = stride

    def forward(self, x):

        resblock = self.resblock (x)

        if self.stride == 1:
            if self.in_channels == self.out_channels:
                resblock  += x

        return resblock

class MyMobileNet(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.preAct = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.layer1 = LinearBottleNeck(32, 16, 1, 1)
        self.layer2 = self.stack_layer(2, 16, 24, 2)
        self.layer3 = self.stack_layer(3, 24, 32, 2)
        self.layer4 = self.stack_layer(4, 32, 64, 2)
        self.layer5 = self.stack_layer(3, 64, 96, 1)
        self.layer6 = self.stack_layer(3, 96, 160, 1)
        self.layer7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

    def forward(self, x):
        x = self.preAct(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def stack_layer(self, repeat, in_features, out_features, stride):

        layers = []
        t = 6
        layers.append(LinearBottleNeck(in_features, out_features, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_features, out_features, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

class MyNetwork(object):
    def getMyMobileNet():
        return MyMobileNet()