# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: base_exp.py
# @Time: 2022/4/18 16:07

from abc import ABCMeta, abstractmethod
import torch
from torch.nn import Module


class BaseExp(metaclass=ABCMeta):
    def __init__(self):
        self.seed = None
        self.output_dir = "./outputs"
        self.print_interval = 100
        self.eval_interval = 10

    @abstractmethod
    def get_model(self) -> Module:
        pass

    @abstractmethod
    def get_data_loader(self, batch_size: int) ->torch.utils.data.DataLoader:
        pass

    @abstractmethod
    def get_optimizer(self, lr: float) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_lr_scheduler(self, **kwargs):
        pass

    @abstractmethod
    def get_loss_func(self):
        pass

    @abstractmethod
    def get_val_loader(self, batch_size: int):
        pass



