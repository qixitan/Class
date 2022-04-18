# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: featuremap.py
# @Time: 2022/3/2 12:28

from torch import nn
import torchvision
from torchvision import transforms
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

net = torchvision.models.vgg11(pretrained=True)
# print(net)
img = cv.imread("..\data\img\img.jpg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = np.array(img)
img = transform(img)
img = img.unsqueeze(0)
print(img.size())

no_of_layers = 0
conv_layers = []
model_children = list(net.children())

for child in model_children:
    if type(child) == nn.Conv2d:
        no_of_layers += 1
        conv_layers.append(child)
    elif type(child) == nn.Sequential:
        for layer in child.children():
            if type(layer) == nn.Conv2d:
                no_of_layers += 1
                conv_layers.append(layer)
print(no_of_layers)

results = [conv_layers[0](img)]
for i in range(1, len(conv_layers)):
    results.append(conv_layers[i](results[-1]))

outputs = results

for num_layer in range(len(outputs)):
    plt.figure(figsize=(50, 10))
    layer_viz = outputs[num_layer][0,:,:,:]
    layer_viz = layer_viz.data
    print("Layer", num_layer+1)
    for i, fiter in enumerate(layer_viz):
        if i==16:
            break
        plt.subplot(2, 8, i+1)
        plt.imshow(fiter, cmap="jet")
        plt.axis("off")
    plt.show()
    plt.close()
