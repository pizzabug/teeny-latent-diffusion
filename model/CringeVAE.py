import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model.cringe.unet import UNet

"""

"""
class CringeVAEModel(pl.LightningModule):
    def __init__(self, dimensions  = [
            32, 64, 128, 256
        ], hparams = None, has_cross_attention = False, img_dim = 512):
        super().__init__()

        self.img_dim = img_dim
        self.vae_module = UNet(dimensions=dimensions, hparams=hparams, has_cross_attention=has_cross_attention)

    def forward(self, x):
        x = self.vae_module(x)
        return x
        
    """
        configure_optimizers

        This is the optimizer for the model.
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer

    """
        training_step
    """
    def training_step(self, train_batch, batch_idx):
        # Grab batch
        y, _ = train_batch

        # Skip if image is None
        if y is None:
            return None

        # Cuda up if needed
        if torch.cuda.is_available():
            y = y.cuda()

        # Forward pass
        y_hat = self.forward(y)
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss)

        # Skip if resulting loss is NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return None

        return loss

    """
        validation_step
    """
    def validation_step(self, val_batch, batch_idx):
        # Grab batch
        y, _ = val_batch

        # Cuda up if needed
        if torch.cuda.is_available():
            y = y.cuda()

        # Forward pass
        y_hat = self.forward(y)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss