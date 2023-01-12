
import torch
import torch.nn as nn
from model.unet.unet import ConvBlock, UpBlock, UNet

"""
    UNet Test
"""
def test_unet():
    # Define a convolutional layer
    conv = UNet()

    # Generate some random input data
    x = torch.randn(1, 3, 512, 512)
    q = torch.randn(1, 3, 512, 512)

    # Apply the convolutional layer to the input data
    y = conv(x, q)

    print(f'Original size: {x.size()}')
    print(f'Convolutional layer output size: {y.size()}')

    if (x.size() == y.size()):
        return True
    
    return False