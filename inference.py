import numpy as np
import torch
from model.CringeLDM import CringeBERTWrapper, CringeLDMWrapper
# Complete import for plt
import matplotlib.pyplot as plt

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

# Load bert
bertWrapper = CringeBERTWrapper()
bertWrapper.loadModel()

# Load the model
modelWrapper = CringeLDMWrapper(bert=bertWrapper, ldm=None)

# Load the image
res = modelWrapper.inference(steps=20, query="I am a cat");

# Convert the image to RGB
res = convert_to_rgb(res)

# Show the image
plt.imshow(res)
