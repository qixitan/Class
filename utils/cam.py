# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: cam.py
# @Time: 2022/4/26 21:03

import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch import nn
import torchvision
from torchvision import transforms

import numpy as np
import cv2 as cv


class Cam(nn.Module):
    def __init__(self, exp, arg):
        super(Cam, self).__init__()
        self.exp = exp
        self.arg = arg
        self.root = "/".join(os.path.abspath(__file__).split("/")[:-2])   # 项目目录
        # print(self.root)
        self.img_root = os.path.join(self.root, "data", "img", self.arg.img)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.feature_layer = self.arg.layer
        self.test_size = self.exp.test_size

        self.model = self.get_model()
        # self.model = torchvision.models.resnet18(pretrained=True)

        # self.model.to(self.device)

        self.gradient = []  # 记录梯度
        self.img_feature = []    # 记录输出特征图

        self.test_transformers = transforms.Compose([
            lambda x: Image.open(x).convert("RGB"),
            transforms.Resize(self.test_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_model(self):
        model = self.exp.get_model().to(self.device)
        model_name = self.arg.exp
        ckpt_path = os.path.join(self.root, "tools", "outputs", model_name, "best_ckpt.pth")
        ckpt = torch.load(ckpt_path)["model"]
        # model = self.load_ckpt(model, ckpt)
        model.load_state_dict(ckpt)
        return model

    def process_img(self, img_root):
        img = self.test_transformers(img_root)
        img = img.unsqueeze(0)
        return img

    def save_grad(self, grad):
        self.gradient.append(grad)

    def getGrad(self, img):
        img = img.to(self.device).requires_grad_(True)
        num = 1
        for name, module in self.model._modules.items():
            if num == 1:
                num += 1
                img_feature = module(img)
                continue
            img_feature = module(img_feature)

            if name == self.feature_layer:
                img_feature.register_hook(self.save_grad)
                self.img_feature.append([img_feature])
            elif name == "avgpool":
                img_feature = img_feature.reshape(img_feature.shape[0], -1)
        # 全连接的输出
        index = torch.max(img_feature, dim=-1)[1]
        one_hot = torch.zeros((1, img_feature.shape[-1]), dtype=torch.float32)
        one_hot[0][index] = 1
        confidenct = torch.sum(one_hot * img_feature.cpu(), dim=1).requires_grad_(True)
        # print(confidenct)
        self.model.zero_grad()
        confidenct.backward(retain_graph=True)
        return self.gradient[-1], self.img_feature[-1][0], img.grad

    def getCam(self, grad_val, img_feature):
        # 对特征图的每个通道进行全局池化
        alpha = torch.mean(grad_val, dim=(2, 3)).cpu()
        img_feature = img_feature.cpu()
        # 将池化后的结果和相应通道特征图相乘
        cam = torch.zeros((img_feature.shape[2], img_feature.shape[3]), dtype=torch.float32)
        for idx in range(alpha.shape[1]):
            cam = cam + alpha[0][idx] * img_feature[0][idx]
        # 进行ReLU操作
        cam = np.maximum(cam.detach().numpy(), 0)
        plt.imshow(cam)
        plt.colorbar()
        plt.savefig(os.path.join(self.save_path, self.arg.img.split(".")[0]+"_org_cam.jpg"))

        # 将cam区域放大到输入图片大小
        cam_ = cv.resize(cam, (224, 224))
        cam_ = cam_ - np.min(cam_)
        cam_ = cam_ / np.max(cam_)
        plt.imshow(cam_)
        plt.savefig(os.path.join(self.save_path, self.arg.img.split(".")[0]+"_scale_cam.jpg"))
        cam = torch.from_numpy(cam)
        return cam, cam_

    def save_cam_with_img(self, cam_, img):
        heatmap = cv.applyColorMap(np.uint8(255*cam_), cv.COLORMAP_JET)
        cam_img = 0.5*heatmap+0.5*np.float32(img)
        cv.imwrite(os.path.join(self.save_path, self.arg.img.split(".")[0]+"_img_with_cam.jpg"), cam_img)

    def __call__(self):
        img_root = self.img_root
        org_img = Image.open(img_root).resize(self.test_size)
        plt.imshow(org_img)
        self.save_path = os.path.join(self.root, "tools", "outputs", self.arg.exp, self.feature_layer)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        plt.savefig(os.path.join(self.save_path, self.arg.img.split(".")[0]+"_org_img.jpg"))
        img = self.process_img(img_root)
        grad_val, img_feature, img_grad = self.getGrad(img)
        cam, cam_ = self.getCam(grad_val, img_feature)
        self.save_cam_with_img(cam_, org_img)
        print("Finish save CAM in {}".format(self.save_path))
        return cam


def cam(exp, arg):
    c = Cam(exp, arg)
    c()
