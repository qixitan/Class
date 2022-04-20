# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: cifar10_base.py
# @Time: 2022/4/18 16:25

from .base_exp import BaseExp
import os
import torch


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        self.num_classes = 5

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 4

        # --------------  training config --------------------- #
        self.max_epoch = 100
        self.basic_lr = 0.1
        self.weight_decay = 5e-4

        self.momentum = 0.9
        self.print_interval = 1
        self.eval_interval = 10
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]   # ?????  好像没用啊

        # -----------------  testing config ------------------ #
        self.test_size = (224, 224)

    def get_model(self):
        from ..ResNet import ResNet18
        self.model = ResNet18(self.num_classes)
        return self.model

    def get_data_loader(self, batch_size: int):
        from model.data.dataset import FlowersDataset
        train_set = FlowersDataset(set_name="flowers", size=self.test_size, train=True)

        # train_sampler = torch.utils.data.SequentialSampler(train_set)
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=self.data_num_workers, pin_memory=True)

        return train_loader

    def get_val_loader(self, batch_size: int):
        from model.data.dataset import FlowersDataset
        val_set = FlowersDataset(set_name="flowers", size=self.test_size, train=False)
        # val_sampler = torch.utils.data.SequentialSampler(val_set)
        val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=self.data_num_workers, pin_memory=True)

        return val_loader

    # def get_evaluator(self, batch_size: int):
    #     from model.evaluators import CIFAR10Evaluator
    #     val_loader = self.get_val_loader(batch_size)
    #     evaluator = CIFAR10Evaluator(dataloader=val_loader)
    #     return evaluator
    #
    # def eval(self, model, evaluator):
    #     return evaluator.evaluate(model)

    def get_optimizer(self, lr: float):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=lr, momentum=self.momentum, weight_decay=self.weight_decay
        )
        return self.optimizer

    def get_lr_scheduler(self):
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epoch)
        return scheduler

    def get_loss_func(self):
        return torch.nn.CrossEntropyLoss()




