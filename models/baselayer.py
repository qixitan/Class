import torch
from torch import nn


def get_activation(act="relu"):
    if act == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    elif act == "silu":
        return nn.SiLU(inplace=True)
    elif act == "relu":
        return nn.ReLU(inplace=True)


def conv3x3(inplanes, planes, stride=1):
    return nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Focus(nn.Module):
    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat((patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,)
        return self.conv(x)


class BaseConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, groups=1, bias=False, act="relu"):
        super(BaseConv, self).__init__()
        self.pad = (kernel_size-1)//2
        self.act = get_activation(act=act)
        self.con = nn.Conv2d(in_channels=inplanes, out_channels=planes, kernel_size=kernel_size, stride=stride,
                             padding=self.pad, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(planes, eps=5e-5, momentum=1e-1)

    def forward(self, x):
        return self.act(self.bn(self.con(x)))

    def fuseforward(self, x):
        return self.bn(self.con(x))


class DWConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, act=None):
        super(DWConv, self).__init__()
        self.dconv = BaseConv(inplanes, inplanes, kernel_size=kernel_size, stride=stride, groups=planes,
                              act=act)
        self.pconv = BaseConv(inplanes, planes, kernel_size=1, stride=1, groups=1, act=act)

    def forward(self, x):
        return self.pconv(self.dconv(x))


class Residual(nn.Module):
    # resnet18、resnet34中使用的基本residual块
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act="relu"):
        super(Residual, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = get_activation(act)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class Bottleneck(nn.Module):
    # 在resnet50、resnet101、resnet152中使用的residual块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act="relu"):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = get_activation(act)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.act(out)

        return out


class InRes(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, act="relu"):
        super(InRes, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.act = get_activation(act)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.act(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))

        if self.downsample is not None:
            residual = self.downsample(x)

        out = residual + out1 + out2
        out = self.act(out)

        return out


# 得再想想
class InResneck(nn.Module):
    # 在resnet50、resnet101、resnet152中使用的InRes块
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, act="relu"):
        super(InResneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=3, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.act = get_activation(act)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.act(self.bn1(self.conv1(x)))

        out2 = self.bn2(self.conv2(out1))

        out2 = out1 + out2

        out3 = self.bn3(self.conv3(out2))

        if self.downsample is not None:
            residual = self.downsample(x)

        out3 = residual + out3
        out3 = self.act(out3)

        return out3  #