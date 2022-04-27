# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: get_cam.py
# @Time: 2022/4/26 20:57

import argparse
from utils import cam
from model.exp import get_exp


def make_parser():
    parser = argparse.ArgumentParser(description=" Get CAM")
    parser.add_argument(
        "-exp", "--expname", dest="exp", type=str, default="ResNet18_flowers", help="model be used to calculate CAM"
    )
    parser.add_argument(
        "-img", "--img_name", dest="img", type=str, default="tulip.jpg", help="the image name in data/img to get CAM"
    )
    parser.add_argument(
        "-l", "--layer", dest="layer", type=str, default="layer4", help=" which layer to calculate in model"
    )

    return parser


def main(exp, arg):
    cam(exp, arg)
    pass


if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp)
    main(exp, args)
