import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch.nn import functional as F

from utils import add_noise
from model.CringeVAE import CringeVAEModel
from model.CringeCLIP import CringeCLIPModel
from model.unet.unet import UNet


class CringeDenoiserModel(pl.LightningModule):
    """
        Denoiser Model

        This is the definition of the LDM denoiser model.
    """

    def __init__(self, hparams=None, vae_model: CringeVAEModel | None = None, diffuser_shapes=[
        32, 64, 128, 256
    ], img_dim=256, clip_model = None):
        super().__init__()
        self.img_dim = img_dim
        self.dropout = 0.02
        self.vae_model: CringeVAEModel = vae_model  # type: ignore
        self.clip_model: CringeCLIPModel = clip_model # type: ignore

        # Diffusion UNet
        self.UNet = UNet(
            dimensions=diffuser_shapes,
            hparams=hparams,
            has_cross_attention=True
        )

        # Image space decoder
        self.imageSpaceDecoder = nn.Sequential(
            nn.Conv2d(3, 6, 12, padding='same'),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.Conv2d(6, 3, 24, padding='same'),
            nn.BatchNorm2d(3),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Conv2d(3, 3, 12, padding='same'),
        )

    def forward(self, q, x=None, steps=20):
        """
            forward

            self: The model
            q: Query tensor from BERT
            x: Image tensor
            steps: Number of steps to denoise the image
        """

        # if torch.cuda.is_available():
            # x = x.cuda()
            # q = q.cuda()

        # Load the image
        if x is None:
            # Generate noise; q's batch dimension is at 1th element
            x = torch.randn(q.shape[1], 3, self.img_dim, self.img_dim)
            x = x.to(q)

        # Put the image through the VAE
        with torch.no_grad():
            x = self.vae_model(x)

        # We denoise for multiple steps
        for i in range(steps):
            # This is the latent space
            x = self.UNet(x, q)
            # Image space decoder
            x = self.imageSpaceDecoder(x)

        return x

    def configure_optimizers(self):
        """
            configure_optimizers

            This is the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        """
            training_step
        """
        # Grab batch
        y, q = train_batch

        # Skip if image is None
        if y is None:
            return None

        # Cuda up if needed
        # if torch.cuda.is_available():
            # y = y.cuda()
            # q = q.cuda()
        
        # Generate x batch, which is a slightly noisier version of y
        x = add_noise(y)

        # Forward pass
        y_hat = self.forward_with_q(query=q, x=x, steps=1)
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
        y, q = val_batch

        # Cuda up if needed
        # if torch.cuda.is_available():
            # y = y.cuda()
            # q = q.cuda()
        
        # Generate x batch, which is a slightly noisier version of y
        x = add_noise(y)

        # Forward pass
        y_hat = self.forward_with_q(q, x=x, steps=1)
        loss = F.l1_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def forward_with_q(self, query, x=None, steps=1):

        # Get the BERT output
        q = torch.tensor(
            self.clip_model.tokenizer(query)).to(self.device)

        # Get q
        with torch.no_grad():
            if q is not None:
                q = self.clip_model.forward(text=q)
            else:
                # Preprocess image and send it through clip. Right now a bit hard, so TODO.
                return None
        
        # Forward pass
        return self.forward(q, x, steps)
