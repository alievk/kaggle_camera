from torch import nn as N
import torch.nn.init as init


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
