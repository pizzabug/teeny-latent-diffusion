# Model Notes

Currently key modules within the LDM are:
- **Denoising UNET** for the bulk of generation
- **Cross Attention** for conditional generation
- **BERT** for text encoding.

## Denoising UNET
### Conv Blocks
Conv blocks are modules comprised of a series of two `Conv2D` layers. They are provided an `in_channel` and an `out_channel` parameter that both dictate input and output shapes.

### Up Blocks
Up block make use of the `ConvTranspose2d` function. This is a transposed convolutional layer that upsamples the input by a factor of `stride`.

```py
x = torch.randn(1, 3, 32, 32)
# Define a transpose convolutional layer
transpose_conv = nn.ConvTranspose2d(3, 3, kernel_size=2, stride=2)
# Upsample the output of the convolutional layer by a factor of 2
z = transpose_conv(x)
```

They also have residual connections `p`. These are concatted and considered during ConvBlock size definitions, with the input channels of the ConvBlock being consistent the shape of the concat of input `x` and the residual `p` (often just `out_channels * 3`, since the `x` would have been transposed).