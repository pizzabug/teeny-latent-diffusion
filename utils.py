import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDM
from PIL import Image

# Inference of UNET model
# Path: inference.py
# Compare this snippet from model/modules/UNetWithCrossAttention.py:

# Convert Conv2d output to RGB image
def convert_to_rgb(x):
    x = x.squeeze(0)
    x = x.permute(1, 2, 0)
    x = x.detach().numpy()
    x = (x - x.min()) / (x.max() - x.min())
    x = (x * 255).astype(np.uint8)
    return x

# Exprt RGB image to PNG
def export_image_to_png(x, path = "out.png"):
    # Export the image to PNG
    img = Image.fromarray(x, 'RGB')
    img.save(path)

# Convert RGB image to Conv2d input
def convert_to_conv2d(x):
    x = x.astype(np.float32)
    x = x / 255
    x = (x - x.min()) / (x.max() - x.min())
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    return x

def import_image_from_path(path = "/mnt/e/Source/unsplash-lite-corpus-preprocess/db/img/xQSLtWJqJ14.png"):
    # Import an image from a file
    img = Image.open(path)
    # Resize the image
    img = img.resize((256, 256))
    # Convert the image to numpy array
    img = np.array(img)
    # Convert the image to Conv2d input
    img = convert_to_conv2d(img)
