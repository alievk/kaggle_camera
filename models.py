from torch import nn as N
import torchvision.models as M
import torch.nn.functional as F


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

    def fresh_parameters(self):
        return self.model.fc.parameters()
