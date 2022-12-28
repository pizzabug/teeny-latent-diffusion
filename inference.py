import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDM
from PIL import Image

from utils import convert_to_rgb, import_image_from_path, export_image_to_png

print ("So you're uh... inferencing.")

# Load the model
model = CringeLDM()
model.load_state_dict(torch.load("checkpoints/model.pt"))

if (torch.cuda.is_available()):
    model = model.cuda()
    print ("Using GPU.")

while (True):
    q = input("> ")


    # Import an image
    img = import_image_from_path()

    # Load the image
    res = model.forward_with_q(query=q, x=img, steps=1);

    # Convert the image to RGB
    res = convert_to_rgb(res)

    # Show the image
    plt.imshow(res)

    export_image_to_png(res, "out.png")