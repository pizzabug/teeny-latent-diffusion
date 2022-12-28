import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDM
from PIL import Image

from utils import convert_to_rgb, import_image_from_path, export_image_to_png

print ("So you're uh... inferencing.")

while (True):
    q = input("> ")

    # Load the model
    modelWrapper = CringeLDM()

    # Import an image
    img = import_image_from_path()

    # Load the image
    res = modelWrapper.forward_with_q(query=q, x=img, steps=1);

    # Convert the image to RGB
    res = convert_to_rgb(res)

    # Show the image
    plt.imshow(res)

    export_image_to_png(res, "out.png")