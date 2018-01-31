from torch import nn as N
import torchvision.models as M
import torch.nn.functional as F
import torch.nn.init as init


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


class SqueezeNet(N.Module):
    def __init__(self, num_classes, model_creator=M.squeezenet1_0, pretrained=True):
        super().__init__()

        model = model_creator(pretrained)
        model.num_classes = num_classes
        final_conv = N.Conv2d(512, num_classes, kernel_size=1)
        model.classifier = N.Sequential(
            N.Dropout(p=0.5),
            final_conv,
            N.ReLU(inplace=True),
            N.AvgPool2d(13, stride=1)
        )
        init.normal(final_conv.weight.data, mean=0.0, std=0.01)

        self.model = model
        self.final_conv = final_conv

    def forward(self, x):
        return self.model(x)

    def fresh_parameters(self):
        return self.final_conv.parameters()
