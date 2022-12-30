import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import (
    BertModel,
    BertTokenizer
)

"""
    Base Convolutional Block

    This is a double convolutional block. It is used in the UNet model.
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hparams = None):
        super(ConvBlock, self).__init__()
        self.dropout = 0.02

        # First convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.02),
            nn.ReLU(inplace=True)
        )
        # Second convolution
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

"""
    Down Convolutional Block

    This is a down convolutional block. It is used in the UNet model. MaxPools.
"""
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x, q):
        x = self.conv(x)
        p = self.maxpool(x)
        return x, p

"""
    Up Convolutional Block
    
    This is an up convolutional block. It is used in the UNet model. Upsamples.
"""
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        # We need extra space for self attention TODO: Add cross attention!
        self.conv = ConvBlock(out_channels + out_channels, out_channels)
    
    def forward(self, x1, x2):
        x = self.up(x1)
        x = torch.cat([x2, x], dim=1)
        return self.conv(x)

"""
    UNet Module


"""