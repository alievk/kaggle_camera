import math

import torch
import torchvision.models as M
import torch.nn.functional as F
import torch.nn as N
import torch.utils.model_zoo as model_zoo


class Bottleneck(N.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = N.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = N.BatchNorm2d(planes)
        self.conv2 = N.Conv2d(planes, planes, kernel_size=3, stride=stride,
                              padding=1, bias=False)
        self.bn2 = N.BatchNorm2d(planes)
        self.conv3 = N.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = N.BatchNorm2d(planes * 4)
        self.relu = N.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class AvgPool(N.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, (x.size(2), x.size(3)))


class ResNet(N.Module):
    def __init__(self, num_classes, model_creator=M.resnet50, pretrained=True):
        super().__init__()

        model = model_creator(pretrained)
        model.avgpool = AvgPool()
        model.fc = N.Linear(model.fc.in_features, num_classes)

        self.model = model

    def forward(self, x):
        return self.model(x)


class ResNet(N.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = N.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
        self.bn1 = N.BatchNorm2d(64)
        self.relu = N.ReLU(inplace=True)
        self.maxpool = N.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = AvgPool()
        self.fc = N.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, N.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, N.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = N.Sequential(
                N.Conv2d(self.inplanes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                N.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return N.Sequential(*layers)

    def set_number_of_classes(self, num_classes):
        self.fc = N.Sequential(
            N.Linear(self.fc.in_features+1, 512),
            N.Dropout(0.3),
            N.Linear(512, 128),
            N.Dropout(0.3),
            N.Linear(128, num_classes)
        )

    def forward(self, x, O):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, O.view(-1, 1)], 1)
        x = self.fc(x)

        return x

    def feature_parameters(self):
        for layer in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
            for name, param in layer.named_parameters():
                yield param

    def classifier_parameters(self):
        return self.fc.parameters()


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://download.pytorch.org/models/resnet50-19c8e357.pth'))
    model.set_number_of_classes(kwargs['num_classes'])
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(
            model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'))
    model.set_number_of_classes(kwargs['num_classes'])
    return model