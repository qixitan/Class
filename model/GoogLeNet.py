# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: GoogLeNet.py
# @Time: 2022/3/1 17:37
import torch
from torch import nn


class BaseConv(nn.Module):
    def __init__(self, inplanes, planes, **kwargs):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, **kwargs)
        self.bn = nn.BatchNorm2d(planes)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Inception(nn.Module):
    # Inception-v2
    def __init__(self, inplanes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        self.b1 = BaseConv(inplanes, n1x1, kernel_size=1)
        # self.b2_1x1_a = BaseConv(inplanes, n3x3red, kernel_size=1)
        # self.b2_3x3_b = BaseConv(n3x3red, n3x3, kernel_size=3, padding=1)
        self.b2 = nn.Sequential(
            BaseConv(inplanes, n3x3red, kernel_size=1),
            BaseConv(n3x3red, n3x3, kernel_size=3, padding=1),
        )
        # self.b3_1x1_a = BaseConv(inplanes, n5x5red, kernel_size=1)
        # self.b3_3x3_b = BaseConv(n5x5red, n5x5, kernel_size=3, padding=1)
        # self.b3_3x3_c = BaseConv(n5x5, n5x5, kernel_size=3, padding=1)
        self.b3 = nn.Sequential(
            BaseConv(inplanes, n5x5red, kernel_size=1),
            BaseConv(n5x5red, n5x5, kernel_size=3, padding=1),
            BaseConv(n5x5, n5x5, kernel_size=3, padding=1),
        )

        # self.b4_pool = nn.MaxPool2d(3, stride=1, padding=1)
        # self.b4_1x1 = BaseConv(inplanes, pool_planes, kernel_size=1)
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            BaseConv(inplanes, pool_planes, kernel_size=1),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1, y2, y3, y4], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, inplanes=3, num_classes=1000):
        super(GoogLeNet, self).__init__()
        self.stem = nn.Sequential(
            BaseConv(inplanes, 64, kernel_size=7, stride=2,  padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            BaseConv(64, 64, kernel_size=1),
            BaseConv(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.b3(self.a3(x))
        x = self.maxpool(x)
        x = self.e4(self.d4(self.c4(self.b4(self.a4(x)))))
        x = self.maxpool(x)
        x = self.b5(self.a5(x))
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

