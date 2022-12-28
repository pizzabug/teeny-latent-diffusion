# To incorporate cross-attention into the forward step of the U-Net, you can pass the output of a BERT encoder as the query input to the cross-attention module at each layer of the U-Net. Here is an example of how you might do this in Python using PyTorch:
import torch
import torch.nn as nn
import transformers
import numpy as np

from model.cringe.cross_attention import CrossAttention

class UNetLayerWithCA(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_channels, query_channels):
        super(UNetLayerWithCA, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_channels, padding='same')
        self.cross_attention = CrossAttention(256, 256, query_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_channels, padding='same')
        self.dropout = torch.nn.Dropout(0.02)

    def forward(self, x, query):
        x = self.conv1(x)
        x = self.cross_attention(x, query)
        x = self.dropout(x)
        x = self.conv2(x)
        return x, query

class UNetWithCrossAttention(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 3, query_channels = 768):
        super(UNetWithCrossAttention, self).__init__()
        
        self.decoderL1 = UNetLayerWithCA(3, 3, 12, query_channels)
        self.decoderL2 = UNetLayerWithCA(3, 6, 24, query_channels)
        self.decoderL3 = UNetLayerWithCA(6, 8, 48, query_channels)

        self.encoderL1 = UNetLayerWithCA(8, 6, 48, query_channels)
        self.encoderL2 = UNetLayerWithCA(6, 3, 24, query_channels)
        self.encoderL3 = UNetLayerWithCA(3, 3, 12, query_channels)

    def forward(self, x, query):
        x, _ = self.decoderL1(x, query)
        x, _ = self.decoderL2(x, query)
        x, _ = self.decoderL3(x, query)

        x, _ = self.encoderL1(x, query)
        x, _ = self.encoderL2(x, query)
        x, _ = self.encoderL3(x, query)
        return x