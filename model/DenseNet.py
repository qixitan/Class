# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: DenseNet.py
# @Time: 2022/3/1 14:35
import torch
from torch import nn


def baseconv(inplanes, planes):
    return nn.Sequential(
        nn.BatchNorm2d(inplanes),
        nn.ReLU(inplace=True),
        nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
    )


class DenseBlock(nn.Module):
    def __init__(self, inplanes, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        block = []
        channel = inplanes
        for i in range(num_layers):
            block.append(baseconv(channel, growth_rate))
            channel += growth_rate
        self.net = nn.Sequential(*block)

    def forward(self, x):
        for layer in self.net:
            out = layer(x)
            x = torch.cat([out, x], dim=1)
        return x


def transition_block(inplanes, planes):
    return nn.Sequential(
        nn.BatchNorm2d(inplanes),
        nn.Conv2d(inplanes, planes, kernel_size=1,),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


class DenseNet(nn.Module):
    def __init__(self, inplanes, block_layers, num_classes, growth_ratio=32):
        super(DenseNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inplanes, 64, kernel_size=7, stride=2, padding=3,),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.denseblock1 = self._make_dense_block(64, growth_ratio, num_layer=block_layers[0])
        self.translayer1 = self._make_transition_layer(256)
        self.denseblock2 = self._make_dense_block(128, growth_ratio, num_layer=block_layers[1])
        self.translayer2 = self._make_transition_layer(512)
        self.denseblock3 = self._make_dense_block(256, growth_ratio, num_layer=block_layers[2])
        self.translayer3 = self._make_transition_layer(1024)
        self.denseblock4 = self._make_dense_block(512, growth_ratio, num_layer=block_layers[3])
        self.global_average = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.denseblock1(x)
        x = self.translayer1(x)
        x = self.denseblock2(x)
        x = self.translayer2(x)
        x = self.denseblock3(x)
        x = self.translayer3(x)
        x = self.denseblock4(x)
        x = self.global_average(x)
        x = x.view(x.shape[0], -1)
        x = self.linear(x)
        return x

    def _make_dense_block(self, channels, growth_rate, num_layer):
        block = [DenseBlock(channels, growth_rate, num_layer)]
        channels += num_layer*growth_rate
        return nn.Sequential(*block)

    def _make_transition_layer(self, channels):
        block = [transition_block(channels, channels//2)]
        return nn.Sequential(*block)


def DenseNet121(num_classes=1000):
    return DenseNet(3, [6, 12, 24, 16], num_classes=num_classes)


def DenseNet161(num_classes=1000):
    return DenseNet(3, [6, 12, 36, 24], num_classes=num_classes)


def DenseNet169(num_classes=1000):
    return DenseNet(3, [6, 12, 32, 32], num_classes=num_classes)


def DenseNet201(num_classes=1000):
    return DenseNet(3, [6, 12, 48, 32], num_classes=num_classes)

