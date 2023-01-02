import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch

from model.CringeDenoiser import CringeDenoiserModel
from model.CringeVAE import CringeVAEModel
from PIL import Image

from utils import convert_to_rgb, import_image_from_path, export_image_to_png

def load_model (vae_model_dir="checkpoints/vae/model.ckpt", denoiser_model_dir="checkpoints/ldm/model.ckpt"):
    # Load checkpoint if it exists 
    vae_model = CringeVAEModel(dimensions=[16,32,64,128])
    if vae_model_dir is not None:
        vae_model.load_state_dict(torch.load(vae_model_dir)["state_dict"])

    # Load the denoiser model
    denoiser_model = CringeDenoiserModel(diffuser_shapes=[16,32,64,128], vae_model=vae_model)
    if denoiser_model_dir is not None:
        denoiser_model.load_state_dict(torch.load(denoiser_model_dir)['state_dict'])

    if (torch.cuda.is_available()):
        denoiser_model = denoiser_model.cuda()
        print ("Using GPU.")

    return denoiser_model

def inference(model, q=""):

    # Import an image
    img = import_image_from_path()

    # Load the image
    res = model.forward_with_q(query=q, x=img, steps=50);

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