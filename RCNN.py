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
from tqdm import tqdm
from torchvision.models import resnet50, vgg19, vgg16
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake

def conv3x3(in_planes, out_planes, stride=1):
    """ 3x3 convolution with padding """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class RcnnBlock(nn.Module):
    ''' Residual block of ResUnet'''
    def __init__(self, inplanes, planes):
        super(RcnnBlock, self).__init__()
        self.conv1 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU()),
        ]))

        self.conv2 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU()),
        ]))
        self.conv3 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU()),
        ]))
        self.conv4 = nn.Sequential(collections.OrderedDict([
            ('conv', nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)),
            ('bn', nn.BatchNorm2d(planes)),
            ('relu', nn.ReLU()),
            ('glbpool', nn.AvgPool2d(kernel_size=2, stride=2)),
        ]))

    def forward(self, x):
        x = self.conv1(x)
        x1 = x
        x = self.conv2(x)
        x = x1 + x
        x = self.conv3(x)
        x = x1 + x
        x = self.conv4(x)
        return x

class RCNN(nn.Module):
    def __init__(self, input_channels, n_classes, patch_size=15,):
        super(RCNN, self).__init__()
        self.model_name = 'RCNN'
        self.input_channels = input_channels
        self.patch_size = patch_size
        # Encoder
        self.layer1 = nn.Sequential(
            conv3x3(input_channels, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # s/1, dim=64, Addtion
        self.layer2 = RcnnBlock(64, 128)

        self.layer3 = RcnnBlock(128, 256)
        self.layer4 = RcnnBlock(256, 512)
        self.flatten = nn.Flatten()
        self.features_sizes = self._get_sizes()
        self.classifier = nn.Linear(self.features_sizes, n_classes)
        #self.fc = nn.Linear(1024, n_classes)

    def _get_sizes(self):
        x = torch.zeros((1, self.input_channels, self.patch_size, self.patch_size))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.flatten(x)
        w, h = x.size()
        size0 = w * h
        return size0

    def forward(self, x):
        x = x.squeeze()
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #x = self.layer4(x)
        x = self.flatten(x)
        x = self.classifier(x)
       # x = self.fc(x)
        return x