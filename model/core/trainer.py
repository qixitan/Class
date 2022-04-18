# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: trainer.py
# @Time: 2022/4/18 18:54


class Trainer:
    def __init__(self, exp, args):
        self.exp = exp
        self.args = args

        # raining related
        self.max_epoch = exp.max_epoch

        # data/dataloader related attr
        self.best_acc = 0


    def train(self):
        self.before_train()


    def before_train(self):
        model = self.exp.get_model()
