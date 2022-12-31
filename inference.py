import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDMModel
from PIL import Image

from utils import convert_to_rgb, import_image_from_path, export_image_to_png

def load_model (model_dir=None):
    # Load the model
    model = CringeLDMModel()
    if model_dir is not None:
        model.load_state_dict(torch.load("checkpoints/ldm/model.ckpt")['state_dict'])

    if (torch.cuda.is_available()):
        model = model.cuda()
        print ("Using GPU.")

    return model

def inference(model, q=""):

    # Import an image
    img = import_image_from_path()

    # Load the image
    res = model.forward_with_q(query=q, x=img, steps=1);

    # Convert the image to RGB
    res = convert_to_rgb(res)

    # Show the image
    plt.imshow(res)

    export_image_to_png(res, "out.png")

def inference_loop():
    print ("So you're uh... inferencing.")
    model = load_model()
    while True:
        q = input("> ")
        inference(model, q)

if __name__ == "__main__":
    inference_loop()