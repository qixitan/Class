# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: Vgg.py
# @Time: 2022/3/1 16:27

import torch.nn as nn
import math


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.classLinear = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        x = self.classLinear(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0]*m.kernel_size[1]*m.out_channels
                m.weight.data.normal_(0, math.sqrt(2./n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, BN=False):
    layer = []
    inplanes = 3
    for v in cfg:
        if v == "M":
            layer += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(inplanes, v, kernel_size=3, padding=1)
            if BN:
                layer += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layer += [conv2d, nn.ReLU(inplace=True)]
            inplanes = v
    return nn.Sequential(*layer)


cfg = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["A"]), num_classes=num_classes, **kwargs)


def vgg11_bn(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["A"], BN=True), num_classes=num_classes, **kwargs)


def vgg13(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["B"]), num_classes=num_classes, **kwargs)


def vgg13_bn(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["B"], BN=True), num_classes=num_classes, **kwargs)


def vgg16(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["C"]), num_classes=num_classes, **kwargs)


def vgg16_bn(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["C"], BN=True), num_classes=num_classes, **kwargs)


def vgg19(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["D"]), num_classes=num_classes, **kwargs)


def vgg19_bn(num_classes=1000, **kwargs):
    return VGG(make_layers(cfg["D"], BN=True), num_classes=num_classes, **kwargs)

