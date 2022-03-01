# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: train_cifar10.py
# @Time: 2022/3/1 14:31
import torch
from torch import nn
import torchvision

from core import train
from model import *
from utils import transform

# 设置超参数 可根据自己服务器自行修改
batch_size, num_workers = 8, 2  # 设置batch_size大小和num_workers数目
num_epochs = 100  # 全部训练epoch
lr = 0.1  # 初始化学习率
num_classes = 10  # 由于训练的为CIFAR10数据集 类别数设为10
net = ResNet18(num_classes)  # 使用其他resnet模型请将修改 若使用vgg请修稿vgg中的classlayer中的512*7*7
stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
net.stem = stem  # 原始resnet的stem缩放太大

# 数据增广
train_transform = transform.classes_transform["train"]
test_transform = transform.classes_transform["val"]
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.4914, 0.4882, 0.4465],
#         std=[0.2023, 0.1994, 0.2010],
#     )
# ])
#
# test_transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.4914, 0.4882, 0.4465],
#         std=[0.2023, 0.1994, 0.2010],
#     )
# ])

# 数据迭代器
train_set = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=train_transform
)
test_set = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=test_transform
)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)

# 开始训练
train(net=net, train_loader=train_loader, test_loader=test_loader, num_epochs=num_epochs, lr=lr, batch_size=batch_size)

