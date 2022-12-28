import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch

from model.CringeLDM import CringeLDM
from PIL import Image

from utils import convert_to_rgb, import_image_from_path

from data.unsplashlite import UnsplashLiteDataset
import torch.utils.data
from torch.utils.data import DataLoader
import pytorch_lightning as pl

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# data
dataset = UnsplashLiteDataset(root_dir='/mnt/e/Source/unsplash-lite-corpus-preprocess/db')
training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

train_loader = DataLoader(training_set, batch_size=32)
val_loader = DataLoader(validation_set, batch_size=32)

# model
model = CringeLDM().to("cuda:0" if torch.cuda.is_available() else "cpu")

# training
trainer = pl.Trainer(accelerator='gpu', precision=16, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
