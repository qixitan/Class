# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: ResNet18_cifar10.py
# @Time: 2022/4/18 18:43

from model.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.max_epoch = 50
        # self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.eval_interval = 5

