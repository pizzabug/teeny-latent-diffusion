
import torch
import torch.nn as nn
from model.cringe.unet import ConvBlock, UpBlock

# Define a convolutional layer
conv = UpBlock(3,3)

# Define a transpose convolutional layer
transpose_conv = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=3)

# Generate some random input data
x = torch.randn(1, 3, 10, 10)
# Generate some random input data
x2 = torch.randn(1, 3, 20, 20)

# Apply the convolutional layer to the input data
y = conv(x, x2)

# Upsample the output of the convolutional layer by a factor of 2
z = transpose_conv(y)

print(f'Original size: {x.size()}')
print(f'Convolutional layer output size: {y.size()}')
print(f'Upsampled output size: {z.size()}')