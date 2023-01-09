import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from model.unet.unet import UNet


class CringeCLIPModel(pl.LightningDataModule):
    """
        CringeCLIP
    """

    def __init__(self, dimensions=[
        32, 64, 128, 256
    ], hparams=None, has_cross_attention=False, img_dim=512):
        super().__init__()

        self.img_dim = img_dim
        self.clip_module = UNet(
            dimensions=dimensions, hparams=hparams, has_cross_attention=has_cross_attention)
