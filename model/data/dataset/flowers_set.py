# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: flowers_set.py
# @Time: 2022/4/20 10:25

import os
import glob
import csv

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image


class Flowers(Dataset):
    def __init__(self, set_name: str, size: (int, int), train: bool = True):
        super(Flowers, self).__init__()
        self.project_root = "/"+"/".join(os.path.abspath(__file__).split("/")[1:6])
        self.root = os.path.join(self.project_root, "data", set_name)
        self.size = size
        self.train = train
        self.name2label = {}
        for name in sorted(os.listdir(os.path.join(self.root))):
            if not os.path.isdir(os.path.join(self.root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        # print(self.name2label)
        # images, labels
        self.images, self.labels = self.load_csv(set_name+".csv")
        if train:  # 80%
            self.images = self.images[:int(0.8*len(self.images))]
            self.labels = self.labels[:int(0.8*len(self.labels))]
        else:
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        # item [0~len(self.images]
        # self.images, self.labels
        # img:/home/atr/Documents/tqx/Classification/data/flowers/dandelion/19617501581_606be5f716_n.jpg
        img, label = self.images[item], self.labels[item]
        train_tf = transforms.Compose([
            lambda x:Image.open(x).convert("RGB"),  # str path -> image data
            transforms.Resize((int(self.size[0]*1.25), int(self.size[1]*1.25))),
            transforms.RandomRotation(10),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        val_tf = transforms.Compose([
            lambda x:Image.open(x).convert("RGB"),  # str path -> image data
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img = train_tf(img) if self.train else val_tf(img)
        label = torch.tensor(label)
        return img, label

    def load_csv(self, filename):

        if not os.path.exists(os.path.join(self.root, filename)):
            images = []
            for name in self.name2label.keys():
                images += glob.glob(os.path.join(self.root, name, "*png"))
                images += glob.glob(os.path.join(self.root, name, "*jpng"))
                images += glob.glob(os.path.join(self.root, name, "*jpg"))
            # print(len(images), images[len(images)-1])
            import random
            random.seed(42)
            random.shuffle(images)
            with open(os.path.join(self.root, filename), mode="w", newline="") as f:
                writer = csv.writer(f)
                for img in images:
                    # img: /home/atr/Documents/tqx/Classification/data/flowers/dandelion/19617501581_606be5f716_n.jpg
                    name = img.split(os.sep)[-2]
                    label = self.name2label[name]
                    writer.writerow([img, label])
                print("writen into csv file:{}".format(filename))

        # read from csv file
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    def denormalize(self, x_hat):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        mean = torch.tensor(mean).unsqueeze(1).unsqueeze(1)
        std = torch.tensor(std).unsqueeze(1).unsqueeze(1)

        x = x_hat*std+mean
        return x


# def main():
#     import visdom
#     viz = visdom.Visdom()
#
#     root = "flowers"
#     db = Flowers(root, (224, 224), True)
#
#     img, label = next(iter(db))  # 拿到第一个样本
#     print("Sample:", img.shape, label)
#     viz.image(db.denormalize(img), win="sample", opts=dict(title=str(label)))
#
#
# if __name__ == '__main__':
#     main()

