# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: transformer.py
# @Time: 2022/3/1 16:18
import torchvision.transforms as transforms

cifar_transform = {
    "train":
        transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4882, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        ]),
    "val":
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4882, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            )
        ])
}

# Imagenet
# mean,val=[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

