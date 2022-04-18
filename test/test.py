# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: test.py
# @Time: 2022/3/1 14:38

import os

import torch
import torchvision
from torchvision import transforms
from model import *


# net = ResNet18()
# x = torch.randn(1, 3, 224, 224)
# y = net(x)
# # print(y.shape)
print()
from model.exp.build import get_exp
m = get_exp("ResNet18-cifar10")
print(m.max_epoch)
print()
