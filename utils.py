import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDM
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint

# Inference of UNET model
# Path: inference.py
# Compare this snippet from model/modules/UNetWithCrossAttention.py:

# Convert Conv2d output to RGB image
def convert_to_rgb(x):
    x = x.squeeze(0)
    x = x.permute(1, 2, 0)
    x = x.detach().to("cpu").numpy()
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

class RegularCheckpoint(ModelCheckpoint):
    def __init__(self, model, period = 1000):
        super().__init__()
        self.model = model
        self.period = period
    
    def save_checkpoint(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # Create a new directory for the checkpoint if it doesnt already exist
        if not os.path.exists(f"checkpoints/{pl_module.global_step}"):
            os.mkdir(f"checkpoints/{pl_module.global_step}")
        
        # Save the model
        ckptPath = f"checkpoints/{pl_module.global_step}/model.ckpt"
        trainer.save_checkpoint("checkpoints/model.ckpt")
        trainer.save_checkpoint(ckptPath)

        # Then go save some outputs!
        test_captions = [
            "lady on a walk",
            "dog sitting",
            "the sea",
            "mountains",
            "houses"
        ]

        for caption in test_captions:
            # Load the image
            res = self.model.forward_with_q(query=caption);

            # Convert the image to RGB
            res = convert_to_rgb(res)

            # Show the image
            plt.imshow(res)

            # Export the image
            export_image_to_png(res, f"checkpoints/{pl_module.global_step}/{caption}.png")
            export_image_to_png(res, f"checkpoints/{caption}.png")
        
    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
        *args, **kwargs) -> None:
        if trainer.global_step % self.period == 0:
            self.save_checkpoint(trainer, pl_module)