# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: cifar10_evaluator.py
# @Time: 2022/4/19 14:44

import torch
import datetime


class CIFAR10Evaluator:
    """
    CIFAR10 classification
    """
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def evaluate(self, model):
        model = model.eval()
        val_corrent = 0
        start_time = datetime.datetime.now()
        for i, data in enumerate(self.dataloader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            outputs = model(img)
            _, pred = torch.max(outputs, 1)
            val_corrent += (pred == label).sum()
        end_time = datetime.datetime.now()
        val_time = (end_time - start_time).seconds / len(self.dataloader.dataset)
        val_acc = val_corrent / len(self.dataloader.dataset) * 100.

        return val_acc, val_time



