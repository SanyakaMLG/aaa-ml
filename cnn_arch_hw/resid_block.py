import torch
import torch.nn as nn
import numpy as np
import os
import random


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Инициализируйте необходимое количество conv слоёв
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding='same')
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding='same')

        # Инициализируйте необходимое количество batchnorm слоёв
        self.batchnorm1 = nn.BatchNorm2d(in_channels)
        self.batchnorm2 = nn.BatchNorm2d(in_channels)

        # Инициализируйте relu слой
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batchnorm2(out)
        out += x
        out = self.relu(out)
        return out