import pytorch_lightning as pl
import torch

from torch.nn import functional as F

from model.unet.unet import UNet


class CringeBERTModel(pl.LightningModule):
    """
        CringeBERTModel

        This is the VAE model. This is used as a prior to the denoiser module.
    """
    def __init__(self, dimensions  = [
            32, 64, 128, 256
        ], hparams = None, has_cross_attention = False, img_dim = 512):
        super().__init__()

        self.img_dim = img_dim
        self.vae_module = UNet(dimensions=dimensions, hparams=hparams, has_cross_attention=has_cross_attention)

    def forward(self, x):
        x = self.vae_module(x)
        return x
        
    def configure_optimizers(self):
        """
            configure_optimizers

            This is the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
            training_step
        """
        # Grab batch
        y, _ = train_batch

        # Skip if image is None
        if y is None:
            return None

        # Forward pass
        y_hat = self.forward(y)
        loss = F.l1_loss(y_hat, y)
        self.log('train_loss', loss)

        # Skip if resulting loss is NaN or Inf
        if torch.isnan(loss) or torch.isinf(loss):
            return None

        return loss

    def validation_step(self, val_batch, batch_idx):
        """
            validation_step
        """
        # Grab batch
        y, _ = val_batch

        # Forward pass
        y_hat = self.forward(y)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
