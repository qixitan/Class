import torch.nn as nn
from baselayer import Focus, BaseConv, DWConv, Residual, Bottleneck, InRes, InResneck


class ResNet(nn.Module):
    """
    stem: 4  BaseConv:2 * MaxPool2d:2
    out2: 4
    out3: 8
    out4: 16
    out5: 32
    """
    def __init__(self, block, layers, num_classes=20):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.stem = nn.Sequential(
            BaseConv(inplanes=3, planes=64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                                  )

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(512*block.expansion, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc(x)
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))   # 调整通道维数 cnv3开始为减半通道数
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


def get_resnet(version="resnet18"):
    if version == "resnet18":
        model = ResNet(Residual, [2, 2, 2, 2])
    elif version == "resnet34":
        model = ResNet(Residual, [3, 4, 6, 3])
    elif version == "resnet50":
        model = ResNet(Bottleneck, [3, 4, 6, 3])
    elif version == "resnet101":
        model = ResNet(Bottleneck, [3, 4, 23, 3])
    elif version == "resnet152":
        model = ResNet(Bottleneck, [3, 4, 36, 3])

    return model


