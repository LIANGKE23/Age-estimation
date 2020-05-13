#################################################### Ke LIANG ##########################################################
import torch.nn as nn
import pretrainedmodels
import pretrainedmodels.utils
import torch.nn.functional as F
from collections import OrderedDict
import torch
import torchvision.models as models
from config import _C as cfg

def get_model(model_name="vgg16", num_classes=101, pretrained="imagenet"):
    if model_name == "Simple_CNN_Model":
        print("Simple CNN Model")
        model = LK_simple(num_classes)
        return model
    elif model_name == "General_Residual_Model":
        print("General Residual Model")
        model = LK_MODEL(num_classes,1)
        if cfg.Transfer == False:
            return model
        elif cfg.Transfer == True:
            print("General Residual + Transfer Model")
            net = model
            model_dict = net.state_dict()
            trained_model = models.resnet18(pretrained=True)
            pretrained_dict = trained_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            return net
    elif model_name == "Modified_Residual_Model":
        print("Modified Residual Model")
        model = LK_MODEL(num_classes,2)
        if cfg.Transfer == False:
            return model
        elif cfg.Transfer == True:
            print("Modified Residual + Transfer Model")
            net = model
            model_dict = net.state_dict()
            trained_model = models.resnet18(pretrained=True)
            pretrained_dict = trained_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            net.load_state_dict(model_dict)
            return net
    else:
        model = pretrainedmodels.__dict__[model_name](pretrained=pretrained)
        dim_feats = model.last_linear.in_features
        model.last_linear = nn.Linear(dim_feats, num_classes)
        model.avg_pool = nn.AdaptiveAvgPool2d(1)
        return model


class LK_BASIC(nn.Module):

    def __init__(self, ch_in, ch_out, stride=(1, 1)):
        super(LK_BASIC, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        return x1

class LK_addition(nn.Module):

    def __init__(self, ch_in, ch_out, stride=(1, 1),downsample = 1):
        super(LK_addition, self).__init__()
        self.downsample = downsample
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.downsample1 = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(ch_out)
        )
        self.downsample2 = nn.Sequential(
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_in, ch_out, kernel_size=(1, 1), stride=stride, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=(3, 3), stride=(1,1),padding = (1,1), bias=False)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.downsample == 1:
            x1 += self.downsample1(x)
        elif self.downsample == 2:
            x1 += self.downsample2(x)
        x1 = F.relu(x1)
        return x1


class LK_MODEL(nn.Module):

    def __init__(self, num_class, whichone):
        super(LK_MODEL, self).__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu0 = nn.ReLU(inplace=True)
        self.maxpool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)

        self.layer1 = nn.Sequential(
            LK_BASIC(64, 64, stride=(1, 1)),
            LK_BASIC(64, 64, stride=(1, 1))
        )

        self.layer2 = nn.Sequential(
            LK_addition(64, 128, stride=(2, 2),downsample = whichone),
            LK_BASIC(128, 128, stride=(1, 1))
        )

        self.layer3 = nn.Sequential(
            LK_addition(128, 256, stride=(2, 2),downsample = whichone),
            LK_BASIC(256, 256, stride=(1, 1))
        )

        self.layer4 = nn.Sequential(
            LK_addition(256, 512, stride=(2, 2),downsample = whichone),
            LK_BASIC(512, 512, stride=(1, 1))
        )

        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.outlayer = nn.Linear(in_features=512, out_features=num_class, bias=True)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)
        x = self.maxpool0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.outlayer(x)
        return x


class LK_simple(nn.Module):

    def __init__(self, num_class):
        super(LK_simple, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size=7, stride=4, padding=0, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu3 = nn.ReLU(inplace=True)

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc1 = nn.Linear(in_features=384, out_features=num_class, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

class Flat_lk(nn.Module):

    def __init__(self):
        super(Flat_lk, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)
########################################################################################################################
