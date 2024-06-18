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
from my_models.involution import Involution2d, Involution3d

class I3DEtAl(nn.Module):
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
        super(I3DEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.involution1 = Involution3d(in_channels=1, out_channels=4, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.involution2 = Involution3d(in_channels=4, out_channels=8, kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0))
        self.involution3 = Involution3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step
        self.flatten = nn.Flatten()

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
            x = self.involution1(x)
            x = self.involution2(x)
            x = self.involution3(x)
            x = self.flatten(x)
            w, h = x.size()
        return w * h

    def forward(self, x):
        x = self.involution1(x)
        x = self.involution2(x)
        x = self.involution3(x)
        x = self.flatten(x)
        x = self.out_finetune(x)
        return x


class I2DEtAl(nn.Module):
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
        super(I2DEtAl, self).__init__()
        self.input_channels = input_channels
        self.patch_size = patch_size

        self.involution1 = Involution2d(in_channels=input_channels, out_channels=64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.involution2 = Involution2d(in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0))
        self.involution3 = Involution2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Considering those large kernel values, I assume they actually merge the
        # 3D tensors at each step
        self.flatten = nn.Flatten()

        self.features_size = self._get_final_flattened_size()

        # The fc1 has 1024 outputs, where dropout was applied after
        # fc1 with a rate of 0.5
        self.out_finetune = nn.Linear(self.features_size, n_classes)

        #self.apply(self.weight_init)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(
                (1, self.input_channels, self.patch_size, self.patch_size)
            )
            x = self.involution1(x)
            x = self.involution2(x)
            x = self.involution3(x)
            x = self.flatten(x)
            w, h = x.size()
        return w * h

    def forward(self, x):
        x = x.squeeze()
        x = self.involution1(x)
        x = self.involution2(x)
        x = self.involution3(x)
        x = self.flatten(x)
        x = self.out_finetune(x)
        return x
