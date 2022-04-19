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
import argparse


# net = ResNet18()
# x = torch.randn(1, 3, 224, 224)
# y = net(x)
# # print(y.shape)
print()
# from model.exp.build import get_exp
# m = get_exp("ResNet18-cifar10")
# print(m.max_epoch)
print()

from torch import distributed as dist


def get_rank() -> int:
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


print(get_rank())

