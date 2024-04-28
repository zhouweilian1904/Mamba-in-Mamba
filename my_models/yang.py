# -*- coding: utf-8 -*-
# Torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.nn import init
# import cv2
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

class Yangnew(nn.Module):
    """
    DEEP FEATURE EXTRACTION AND CLASSIFICATION OF HYPERSPECTRAL IMAGES BASED ON
                        CONVOLUTIONAL NEURAL NETWORKS
    """

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def __init__(self, input_channels, n_classes, patch_size):
        super(Yangnew, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        a = 17 * input_channels
        c = (input_channels // 2)
        b = 33 * c

        # An input image of size 263x263 pixels is fed to conv1
        # with 96 kernels of size 6x6x96 with a stride of 2 pixels
        self.conv3_1 = nn.Conv3d(1, 16, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv2_1 = nn.Conv2d(input_channels, input_channels, (3, 3), stride=(1, 1), padding=(1, 1))
        # self.conv1_bn = nn.BatchNorm3d(16)
        # self.pool1 = nn.MaxPool3d((4, 2, 2))
        #  3d_2
        self.conv3_2 = nn.Conv3d(17, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3_2_bn = nn.BatchNorm3d(32)
        self.pool3_2 = nn.MaxPool3d((2, 2, 2))
        # 2d_2
        self.conv2_2 = nn.Conv2d(a, c, (3, 3), padding=(1, 1))
        self.conv2_2_bn = nn.BatchNorm2d(c)
        self.pool2_2 = nn.MaxPool2d((2, 2))

        # 3d_3
        self.conv3_3 = nn.Conv3d(33, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.conv3_3_bn = nn.BatchNorm3d(64)
        self.pool3_3 = nn.MaxPool3d((2, 2, 2))

        # 2d_3
        self.conv2_3 = nn.Conv2d(b, 1024, (1, 1))
        self.conv2_3_bn = nn.BatchNorm2d(1024)
        self.pool2_3 = nn.MaxPool2d((2, 2))
        self.flatten = nn.Flatten()

        self.features_size3 = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.fc1 = nn.Linear(self.features_size3, 1024)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, n_classes)

        self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_channels,
                             self.patch_size, self.patch_size))
            x1 = torch.zeros((1, self.input_channels,
                              self.patch_size, self.patch_size))
            x = F.relu((self.conv3_1(x)))
            b, t, c, w, h = x.size()
            x = x.view(b, t, c, w, h)

            x1 = F.relu((self.conv2_1(x1)))
            _, c1, w1, h1 = x1.size()
            x2_1 = x.view(b, t * c, w, h)
            x1_1 = x1.view(b, 1, c1, w1, h1)

            x_new = torch.cat([x, x1_1], dim=1)
            x2_new = torch.cat([x2_1, x1], dim=1)

            x = F.relu(self.conv3_2_bn(self.conv3_2(x_new)))
            x = self.pool3_2(x)
            b, t, c, w, h = x.size()
            x = x.view(b, t, c, w, h)
            x1 = F.relu(self.conv2_2_bn(self.conv2_2(x2_new)))
            x1 = self.pool2_2(x1)
            _, c1, w1, h1 = x1.size()
            x2_1 = x.view(b, t * c, w, h)
            x1_1 = x1.view(b, 1, c1, w1, h1)

            x_new = torch.cat([x, x1_1], dim=1)
            x2_new = torch.cat([x2_1, x1], dim=1)

            x = F.relu(self.conv3_3_bn(self.conv3_3(x_new)))
            x = self.pool3_3(x)

            x1 = F.relu(self.conv2_3_bn(self.conv2_3(x2_new)))
            x1 = self.pool2_3(x1)
            x1 = self.flatten(x1)
            x = self.flatten(x)


            w, h = x1.size()
            size2 = w * h
            w, h = x.size()
            size1 = w * h

            return size2 + size1

    def forward(self, x):
        x = x.view(-1, 1, self.input_channels, self.patch_size, self.patch_size)
        x1 = x.squeeze(1)
        x = F.relu((self.conv3_1(x)))
        x1 = F.relu((self.conv2_1(x1)))
        # print('x1',x1.shape)
        b1, c1, w1, h1 = x1.size()
        b, t, c, w, h = x.size()
        x = x.view(b, t, c, w, h)
        x2 = x.view(b, t * c, w, h)
        x3 = x1.view(b1, 1, c1, w1, h1)
        x1_new = torch.cat([x1, x2], dim=1)
        x2_new = torch.cat([x3, x], dim=1)

        x = F.relu(self.conv3_2_bn(self.conv3_2(x2_new)))
        x = self.pool3_2(x)
        b, t, c, w, h = x.size()
        x = x.view(b, t, c, w, h)
        x2 = x.view(b, t * c, w, h)
        x1 = F.relu(self.conv2_2_bn(self.conv2_2(x1_new)))
        x1 = self.pool2_2(x1)
        b1, c1, w1, h1 = x1.size()
        x3 = x1.view(b1, 1, c1, w1, h1)
        x1_new = torch.cat([x1, x2], dim=1)
        x2_new = torch.cat([x3, x], dim=1)

        x = F.relu(self.conv3_3_bn(self.conv3_3(x2_new)))
        x = self.pool3_3(x)
        x1 = F.relu(self.conv2_3_bn(self.conv2_3(x1_new)))
        x1 = self.pool2_3(x1)
        x1 = self.flatten(x1)
        x = self.flatten(x)
        x_new = torch.cat([x, x1], dim=1)
        x_new = self.fc1(x_new)
        x_new = self.dropout(x_new)
        x_new = self.fc2(x_new)
        return x_new