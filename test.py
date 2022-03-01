# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: test.py
# @Time: 2022/3/1 14:38

import torch
from model import *
net = GoogLeNet()
x = torch.randn(1, 3, 224, 224)
y = net(x)
print(y.shape)
