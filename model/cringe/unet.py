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

from model.cringe.cross_attention import CrossAttention

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
        self.conditional = ConditionalEncoder(out_channels, out_channels)

    def forward(self, x, q):
        x = self.conv(x)
        # TODO: Actually concat q next time
        x = self.conditional(x, q)
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
        self.conditional = ConditionalEncoder(out_channels, out_channels)
        # We need extra space for self attention and up conv TODO: Add cross attention!
        self.conv = ConvBlock(out_channels * 3, out_channels)
    
    def forward(self, x, p, q):
        x = self.up(x)
        # TODO: Actually concat q next time
        x = self.conditional(x, q)
        x = torch.cat([p, x], dim=1)
        return self.conv(x)

"""
    Conditional Encoder

    Encodes cross-attention embeddings.
    TODO: Currently a lil hacky embed 
"""
class ConditionalEncoder (nn.Module):
    def __init__(self, in_channels, out_channels, query_channels = 512):
        super(ConditionalEncoder, self).__init__()
        self.conv = ConvBlock(in_channels*2, out_channels)

    def forward(self, x, q):
        # Crop q if it has more channels than x. Pad the channels if no.
        if q.shape[1] > x.shape[1]:
            q = q[:, :x.shape[1], :, :]
        elif q.shape[1] < x.shape[1]:
            q = F.pad(q, (0, 0, 0, 0, 0, x.shape[1] - q.shape[1]))

        # Crop q if its shape is larger than x, pad with zeroes if smaller
        if q.shape[2] > x.shape[2]:
            q = q[:, :, :x.shape[2], :x.shape[3]]
        elif q.shape[2] < x.shape[2]:
            q = F.pad(q, (0, x.shape[3] - q.shape[3], 0, x.shape[2] - q.shape[2]), "constant", 0)

        # Crop q if its shape is larger than x, pad with zeroes if smaller
        if q.shape[3] > x.shape[3]:
            q = q[:, :, :, :x.shape[3]]
        elif q.shape[3] < x.shape[3]:
            q = F.pad(q, (0, x.shape[3] - q.shape[3], 0, 0), "constant", 0)

        # Cat x and q
        nx = torch.cat([x, q], dim=1)
        nx = self.conv(nx)
        return nx

"""
    UNet Module


"""
class UNet(nn.Module):
    def __init__(self, dimensions  = [
            32, 64, 128, 256
        ], hparams = None):
        super(UNet, self).__init__()

        self.dimensions = dimensions
        self.dropout = 0.02

        # First Convolution
        self.first_conv = ConvBlock(3, self.dimensions[0])

        # Encoder
        self.down1 = DownBlock(self.dimensions[0], self.dimensions[1])
        self.down2 = DownBlock(self.dimensions[1], self.dimensions[2])
        self.down3 = DownBlock(self.dimensions[2], self.dimensions[3])

        # Decoder
        self.up1 = UpBlock(self.dimensions[3], self.dimensions[2])
        self.up2 = UpBlock(self.dimensions[2], self.dimensions[1])
        self.up3 = UpBlock(self.dimensions[1], self.dimensions[0])

        # Final Convolution
        self.final_conv = ConvBlock(self.dimensions[0], 3)

    def forward(self, x, q):
        # First Convolution
        x = self.first_conv(x)

        # Encoder
        x1, x = self.down1(x, q)
        x2, x = self.down2(x, q)
        x3, x = self.down3(x, q)

        # Decoder
        x = self.up1(x, x3, q)
        x = self.up2(x, x2, q)
        x = self.up3(x, x1, q)

        # Final Convolution
        x = self.final_conv(x)
        return x
