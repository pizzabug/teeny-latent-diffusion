import os
import pytorch_lightning as pl
import torch
import torch.utils.data

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from model.CringeLDM import CringeLDM
from data.unsplashlite import UnsplashLiteDataset
from utils import RegularCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# data
dataset = UnsplashLiteDataset(root_dir='/mnt/e/Source/unsplash-lite-corpus-preprocess/db')
training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

train_loader = DataLoader(training_set, batch_size=1)
val_loader = DataLoader(validation_set, batch_size=1)


# model
model = CringeLDM().to("cuda:0" if torch.cuda.is_available() else "cpu")

# Logger
logger = TensorBoardLogger("tb_logs", name="cringeldm")

trainer = pl.Trainer(accelerator='gpu', precision=16, limit_train_batches=0.5, callbacks=[RegularCheckpoint(250),], logger=logger)

# Load checkpoint if it exists
if (os.path.exists("checkpoints/model.ckpt")):
    trainer.fit(model, train_loader, val_loader, ckpt_path="checkpoints/model.ckpt")
else:
    trainer.fit(model, train_loader, val_loader)

