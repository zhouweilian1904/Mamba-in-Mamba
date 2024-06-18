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

class ConvEtAl(torch.nn.Module):
    def __init__(self, input_channels, flatten=True):
        super(ConvEtAl, self).__init__()
        self.feature_size = 64
        self.name = "conv4"

        self.layer1 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(64)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer2 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(128)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer3 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)),
          ('bn',      nn.BatchNorm2d(256)),
          ('relu',    nn.ReLU()),
          ('avgpool', nn.AvgPool2d(kernel_size=2, stride=2))
        ]))

        self.layer4 = nn.Sequential(collections.OrderedDict([
          ('conv',    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False)),
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

class TESTEtAl(nn.Module):
    """
    A semi-supervised convolutional neural network for hyperspectral image classification
    Bing Liu, Xuchu Yu, Pengqiang Zhang, Xiong Tan, Anzhu Yu, Zhixiang Xue
    Remote Sensing Letters, 2017
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size=15, pool_size=None):
        super(TESTEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size
        self.aux_loss_weight = 1

        # "W1 is a 3x3xB1 kernel [...] B1 is the number of the output bands for the convolutional
        # "and pooling layer" -> actually 3x3 2D convolutions with B1 outputs
        # "the value of B1 is set to be 80"
        # resnet = resnet50(pretrained=True)
        # resnet_feature = list(resnet.children())[:-2]
        self.feature = ConvEtAl(input_channels, flatten=True)
        self.features_sizes = self._get_sizes()
        self.classifier = nn.Linear(self.features_sizes, n_classes)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = self.feature(x)
        w, h = x.size()
        size0 = w * h
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.feature(x)
        x = self.classifier(x)
        return x