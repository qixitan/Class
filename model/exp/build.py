# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: build.py
# @Time: 2022/4/18 14:48

import importlib


def get_exp(exp_name):
    # module_name = ".".join(["model, ResNet", exp], )
    # exp_object = importlib.import_module("model")
    # net = getattr(importlib.import_module("model"), args.exp)
    exp = exp_name.replace("-", "_")  # convert string like "ResNet18-cifar10" to "ResNet18_cifar10"
    module_name = ".".join(["exps", "default", exp])
    exp_object = importlib.import_module(module_name).Exp()
    return exp_object
