import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torchvision

from model.CringeLDM import CringeDenoiserModel
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
    # x = (x - x.min()) / (x.max() - x.min())
    x = (x * 255).astype(np.uint8)
    return x

# Exprt RGB image to PNG


def export_image_to_png(x, path="out.png"):
    # Export the image to PNG
    img = Image.fromarray(x, 'RGB')
    img.save(path)

# Convert RGB image to Conv2d input


def convert_to_tensor(x):
    x = x.astype(np.float32)
    x = x / 255.0
    # x = (x - x.min()) / (x.max() - x.min())
    x = torch.from_numpy(x)
    x = x.permute(2, 0, 1)
    x = x.unsqueeze(0)
    return x


def import_image_from_path(path="/mnt/e/Source/unsplash-lite-corpus-preprocess/db/img/xQSLtWJqJ14.png", img_dim=256):
    # Import an image from a file
    img = Image.open(path)
    # Resize the image
    img = img.resize((img_dim, img_dim))
    # Convert the image to numpy array
    img = np.array(img)
    # Convert the image to Conv2d input
    img = convert_to_tensor(img)
    return img


def train_save_checkpoint(steps, trainer, model, checkpoint=False, base_dir="checkpoints/ldm"):
    # Save the model
    ckptPath = f"{base_dir}/{steps}/model.ckpt"
    trainer.save_checkpoint(f"{os.getcwd()}/{base_dir}/model.ckpt")

    if checkpoint:
        # Create a new directory for the checkpoint if it doesnt already exist
        if not os.path.exists(f"{os.getcwd()}/{base_dir}/{steps}"):
            os.mkdir(f"{os.getcwd()}/{base_dir}/{steps}")

        # Copy the model to the new directory
        os.system(
            f"cp {os.getcwd()}/{base_dir}/model.ckpt {os.getcwd()}/{ckptPath}")


def train_save_image_with_q_denoiser(steps, trainer, model, checkpoint=False, base_dir="checkpoints/ldm"):
    # Then go save some outputs!
    test_captions = [
        "lady on a walk",
        "dog sitting",
        "the sea",
        "mountains",
        "houses"
    ]

    images_to_log = []
    with torch.no_grad():
        for caption in test_captions:
            # Load the image
            res = model.forward_with_q(query=caption, steps=20)
            images_to_log.append(res[0])

            # Convert the image to RGB
            res = convert_to_rgb(res)

            # Show the image
            plt.imshow(res)

            # Export the image
            if checkpoint:
                export_image_to_png(
                    res, f"{base_dir}/{steps}/sample {caption}.png")
            export_image_to_png(res, f"{base_dir}/sample {caption}.png")

    grid = torchvision.utils.make_grid(images_to_log)
    model.logger.experiment.add_image(
        f"lady on a walk, dog sitting, the sea, mountains, houses", grid, steps)


def train_save_image_with_img(steps, trainer, model, checkpoint=False, base_dir="checkpoints/ldm"):
    # Then go save some outputs!
    test_image_paths = [
        "checkpoints/vae/reference/1.png",
        "checkpoints/vae/reference/2.png",
        "checkpoints/vae/reference/3.png",
    ]

    images_to_log = []
    with torch.no_grad():
        for path in test_image_paths:
            # Load the image
            img = import_image_from_path(path, model.img_dim)

            # Inference
            if (torch.cuda.is_available()):
                img = img.cuda()

            res = model.forward(x=img)
            images_to_log.append(res[0])

            # Convert the image to RGB
            res = convert_to_rgb(res)

            # Show the image
            plt.imshow(res)

            # Export the image
            if checkpoint:
                export_image_to_png(res, path.replace("reference", f"{steps}"))
            export_image_to_png(res, path.replace("reference/", ""))

    grid = torchvision.utils.make_grid(images_to_log)
    model.logger.experiment.add_image(
        f"lady on a walk, dog sitting, the sea, mountains, houses", grid, steps)


class RegularCheckpoint(ModelCheckpoint):
    def __init__(self, model, period=1000, dump=50, base_dir="checkpoints/ldm", do_q=True, do_img=False):
        super().__init__()
        self.model = model
        self.period = period
        self.dump = dump
        self.base_dir = base_dir
        self.do_q = do_q
        self.do_img = do_img

    def save_checkpoint(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", checkpoint) -> None:
        # Create a new directory for the checkpoint if it doesnt already exist
        train_save_checkpoint(pl_module.global_step, trainer=trainer,
                              model=self.model, checkpoint=checkpoint, base_dir=self.base_dir)
        # Save some samples!
        if self.do_q:
            train_save_image_with_q_denoiser(
                pl_module.global_step, trainer=trainer, model=self.model, checkpoint=checkpoint, base_dir=self.base_dir)
        if self.do_img:
            train_save_image_with_img(pl_module.global_step, trainer=trainer,
                                      model=self.model, checkpoint=checkpoint, base_dir=self.base_dir)

    def on_train_batch_end(
            self, trainer: "pl.Trainer", pl_module: "pl.LightningModule",
            *args, **kwargs) -> None:
        if trainer.global_step % self.dump == 0:
            self.save_checkpoint(trainer, pl_module,
                                 (trainer.global_step % self.period == 0))
