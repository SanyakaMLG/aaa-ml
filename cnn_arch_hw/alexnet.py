import torch
import torch.nn as nn
import numpy as np
import os
import random


class AlexNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Инициализируйте необходимое количество conv слоёв
        self.conv1 = nn.Conv2d(in_channels, 96, 11, stride=4, padding=2)
        self.conv2 = nn.Conv2d(96, 256, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, 3, stride=1, padding=1)

        # Инициализируйте maxpool и relu слои
        self.maxpool = nn.MaxPool2d(3, stride=2)
        self.relu = nn.ReLU()

        # Инициализируйте необходимое количество linear слоёв
        self.linear1 = nn.Linear(6 * 6 * 256, 100)
        self.linear2 = nn.Linear(100, 100)
        self.linear3 = nn.Linear(100, num_classes)

    def forward(self, x):
        # Последовательно примените слои сети
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.flatten(1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x