# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: cifar10_base.py
# @Time: 2022/4/18 16:25

from .base_exp import BaseExp
import os
import torchvision
import torch


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 1000
        self.model = None

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4

        # --------------  training config --------------------- #
        self.max_epoch = 300
        self.basic_lr = 0.1
        self.weight_decay = 5e-4
        self.momentum = 0.9
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]   # ?????  好像没用啊
        self.optimizer = None

        # -----------------  testing config ------------------ #

    def get_model(self):
        from ..ResNet import ResNet18
        self.model = ResNet18(self.num_classes)
        return self.model

    def get_data_loader(self, batch_size: int):
        from model.data import transform
        data_file = "/"+"/".join(os.path.abspath(__file__).split("/")[1:6]) + "/data"
        print(data_file)
        train_set = torchvision.datasets.CIFAR10(root=data_file, train=True,
                                                 transform=transform.cifar_transform["train"])
        val_set = torchvision.datasets.CIFAR10(root=data_file, train=False,
                                                 transform=transform.cifar_transform["val"])
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=self.data_num_workers
        )
        test_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=self.data_num_workers
        )
        return train_loader, test_loader

    def get_optimizer(self, lr: int):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
        )
        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        return scheduler

    def get_evaluator(self):
        pass


