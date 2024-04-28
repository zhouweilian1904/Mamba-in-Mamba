# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
import cv2
import matplotlib.pyplot as plt
# utils
import math
import os
import datetime
import numpy as np
import joblib
import collections
import torchvision
from pylab import *

class Conv3(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(Conv3, self).__init__()
        self.feature_size = 64
        self.name = "conv3"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(1, 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(4)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(4, 4, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(4, 8, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(8)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(8, 8, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(8, 16, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))),
          ('bn',      nn.BatchNorm3d(16)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.Conv3d(16, 16, (3, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1))),
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(512)),
          ('relu',    nn.ReLU()),
          #('avgpool', nn.AvgPool2d(kernel_size=4))
          ('glbpool', nn.AdaptiveAvgPool2d(1))
        ]))

        self.is_flatten = flatten
        self.flatten = nn.Flatten()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
      #  h = self.layer4(h)
        #print(h.size())
        if(self.is_flatten): h = self.flatten(h)
        return h

class C3DEtAl(nn.Module):
    """
    HYPERSPECTRAL CNN FOR IMAGE CLASSIFICATION & BAND SELECTION, WITH APPLICATION
    TO FACE RECOGNITION
    Vivek Sharma, Ali Diba, Tinne Tuytelaars, Luc Van Gool
    Technical Report, KU Leuven/ETH Zürich
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size):
        super(C3DEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        # self.conv1 = nn.Conv3d(1, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.conv1_bn = nn.BatchNorm3d(64)
        # self.pool1 = nn.MaxPool3d((2, 2, 2))
        # #  256 kernels of size 3x3x256 with a stride of 2 pixels
        # self.conv2 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1,1, 1), padding=(1, 1, 1))
        # self.conv2_bn = nn.BatchNorm3d(128)
        # self.pool2 = nn.MaxPool3d((2, 2, 2))
        # # 512 kernels of size 3x3x512 with a stride of 1 pixel
        # self.conv3 = nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # self.conv3_bn = nn.BatchNorm3d(256)
        # self.pool3 = nn.MaxPool3d((2, 2, 2))
        self.feature = Conv3(input_channels, flatten=True)
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.out_finetune = nn.Linear(self.features_size, n_classes)

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, 1, self.input_channels, self.patch_size, self.patch_size)
            )
          #   x = F.relu(self.conv1_bn(self.conv1(x)))
          #   x = self.pool1(x)
          #   b, t, c, w, h = x.size()
          #  # x = x.view(b, 1, t * c, w, h)
          #   x = F.relu(self.conv2_bn(self.conv2(x)))
          #   x = self.pool2(x)
          #   b, t, c, w, h = x.size()
          # #  x = x.view(b, 1, t * c, w, h)
          #   x = F.relu(self.conv3_bn(self.conv3(x)))
          #   x = self.pool3(x)
            x = self.feature(x)
            w, h = x.size()
        return w * h

    def forward(self, x):
        # x = F.relu(self.conv1_bn(self.conv1(x)))
        #         # x = self.pool1(x)
        #         # b, t, c, w, h = x.size()
        #         # x = F.relu(self.conv2_bn(self.conv2(x)))
        #         # x = self.pool2(x)
        #         # b, t, c, w, h = x.size()
        #         # x = F.relu(self.conv3_bn(self.conv3(x)))
        #         # x = self.pool3(x)
        #         # x = x.view(-1, self.features_size)
        x = self.feature(x)
        x = self.out_finetune(x)
        return x