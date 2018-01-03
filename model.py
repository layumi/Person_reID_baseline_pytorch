import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models


######################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_out')
        init.constant(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal(m.weight.data, std=0.001)
        init.constant(m.bias.data, 0.0)

class ft_net(nn.Module):

    def __init__(self, class_num ):
        super().__init__()
        model_ft = models.resnet50(pretrained=True)

        # avg pooling to global pooling
        model_ft.avgpool = nn.AdaptiveAvgPool2d((1,1))

        num_ftrs = model_ft.fc.in_features
        add_block = []
        num_bottleneck = 512
        add_block += [nn.Linear(num_ftrs, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.LeakyReLU(0.1)]
        add_block += [nn.Dropout(p=0.5)]  #default dropout rate 0.5
        #transforms.CenterCrop(224),
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        model_ft.fc = add_block
        self.model = model_ft

        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)
        self.classifier = classifier

    def forward(self, x):
        x = self.model(x)
        x = self.classifier(x)
        return x

