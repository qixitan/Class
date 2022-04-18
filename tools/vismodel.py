# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: vismodel.py
# @Time: 2022/3/1 21:33

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from model import ResNet18
x1 = Variable(torch.randn(1, 3, 224, 224))
x2 = Variable(torch.randn(1, 3, 224, 224))
net = ResNet18(num_classes=10)

with SummaryWriter(comment="ResNet18") as w:
    w.add_graph(net, x1)

