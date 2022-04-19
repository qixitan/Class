# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: train.py
# @Time: 2022/4/18 13:55

from loguru import logger
import argparse
from model.exp import build
from model.core import Trainer


def make_parser():
    parser = argparse.ArgumentParser(description=" Classification model")
    # parser.add_argument(
    #     "-l", "--learn-rate", dest="lr", type=float, default=0.1
    # )
    # parser.add_argument(
    #     "-e", "--epoch", dest="epoch", type=int, default=20, help="number of epoch"
    # )
    # parser.add_argument(
    #     "-nb", "--number-classes", dest="num_classes", type=int, default=10, help="number of classes of data"
    # )
    parser.add_argument(
        "-exp", "--expname", dest="exp", type=str, default="ResNet18_cifar10", help="model is used in train"
    )
    parser.add_argument(
        "-b", "--batch-size", dest="batch_size", type=int, default=256, help="batch size of your date"
    )

    return parser


@logger.catch
def main(exp, args):
    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = build.get_exp(args.exp)
    main(exp, args)
