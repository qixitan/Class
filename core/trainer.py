# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author: qixitan
# @Email: qixitan@qq.com
# @FileName: trainer.py
# @Time: 2022/3/1 14:24
import torch
import torch.nn as nn

import torch.backends.cudnn as cudnn
import os
import datetime


def train(net, train_loader, test_loader, num_epochs, lr, batch_size):
    net = nn.DataParallel(net)
    cudnn.benchmark = True
    best_acc = 0  # 用以保存最好的模型结果
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    for epoch in range(num_epochs):
        net.train()
        train_loss, train_corrent, test_corrent = 0, 0, 0
        start_time = datetime.datetime.now()
        for i, data in enumerate(train_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            optimizer.zero_grad()
            outputs = net(img)
            loss = loss_func(outputs, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = torch.max(outputs, 1)
            train_corrent += (pred==label).sum()
        end_time = datetime.datetime.now()
        train_batch_time = (end_time-start_time).seconds / len(train_loader.dataset) * batch_size # 每个batch_size训练时间
        train_acc = train_corrent / len(train_loader.dataset) * 100.
        #############测试################
        net.eval()
        start_time = datetime.datetime.now()
        for i, data in enumerate(test_loader):
            img, label = data
            img, label = img.cuda(), label.cuda()
            outputs = net(img)
            _, pred = torch.max(outputs, 1)
            test_corrent += (pred==label).sum()
        end_time = datetime.datetime.now()
        test_batch_time = (end_time-start_time).seconds / len(test_loader.dataset) * batch_size
        test_acc = test_corrent / len(test_loader.dataset) * 100.
        scheduler.step()
        print("epoch:{}|{}: loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f}, time: {:.3f}|{:.3f}".format(epoch+1, num_epochs, train_loss, train_acc, test_acc, train_batch_time, test_batch_time))
        if test_acc > best_acc:
            state = {
                "net": net.state_dict(),
                "best_acc": test_acc,
                "epoch": epoch
            }
            if not os.path.isdir("cifar10_resnet_checkpoint"):
                os.mkdir("cifar10_resnet_checkpoint")
            torch.save(state, "./cifar10_resnet_checkpoint/best_ckpt.pth")
            best_acc = test_acc






