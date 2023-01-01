import os
import sys
import pytorch_lightning as pl
import torch
import torch.utils.data

from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.dirtycollate import dirty_collate

from model.CringeLDM import CringeDenoiserModel
from model.CringeVAE import CringeVAEModel
from data.unsplashlite import UnsplashLiteDataset
from utils import RegularCheckpoint, train_save_checkpoint

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

def train_denoiser():
    # hparams while i'm working on it
    img_dim = 512

    # data
    dataset = UnsplashLiteDataset(root_dir='/mnt/e/Source/unsplash-lite-corpus-preprocess/db', img_dim=img_dim)
    training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

    train_loader = DataLoader(training_set, batch_size=1, collate_fn=dirty_collate)
    val_loader = DataLoader(validation_set, batch_size=1, collate_fn=dirty_collate)

    # Load checkpoint if it exists 
    vae_model = CringeVAEModel(dimensions=[16,32,64,128]).to("cuda:0" if torch.cuda.is_available() else "cpu")
    if (os.path.exists("checkpoints/vae/model.ckpt")):
        vae_model.load_state_dict(torch.load("checkpoints/vae/model.ckpt")["state_dict"])
    
    # Load checkpoint if it exists 
    denoiser_model = CringeDenoiserModel(vae_model=vae_model, diffuser_shapes=[16,32,64,128], img_dim=img_dim).to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Logger
    denoiser_logger = TensorBoardLogger("tb_logs", name="cringeldm")

    denoiser_trainer = pl.Trainer(
        accelerator='gpu', 
        precision=16, 
        limit_train_batches=0.5, 
        callbacks=[
            RegularCheckpoint(
                model=denoiser_model, 
                period=250, 
                base_dir="checkpoints/ldm",
                do_q=True,
                do_img=False,
            ),
        ], 
        logger=denoiser_logger)
    while True:
        try:
            # Load checkpoint if it exists 
            if (os.path.exists("checkpoints/ldm/model.ckpt")):
                denoiser_trainer.fit(denoiser_model, train_loader, val_loader, ckpt_path="checkpoints/ldm/model.ckpt")
            else:
                denoiser_trainer.fit(denoiser_model, train_loader, val_loader)
        except Exception as e:
            tb = sys.exc_info()[2]
            print(e.with_traceback(tb))
        

def train_vae():
    # hparams while i'm working on it
    img_dim = 512

    # data
    dataset = UnsplashLiteDataset(root_dir='/mnt/e/Source/unsplash-lite-corpus-preprocess/db', img_dim=img_dim)
    training_set, validation_set = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), int(len(dataset)*0.2)])

    train_loader = DataLoader(training_set, batch_size=8, collate_fn=dirty_collate)
    val_loader = DataLoader(validation_set, batch_size=8, collate_fn=dirty_collate)

    # Load checkpoint if it exists 
    vae_model = CringeVAEModel(dimensions=[16,32,64,128]).to("cuda:0" if torch.cuda.is_available() else "cpu")

    # Logger
    vae_logger = TensorBoardLogger("tb_logs", name="cringeldmvae")

    vae_trainer = pl.Trainer(
        accelerator='gpu', 
        precision=16, 
        limit_train_batches=0.5, 
        callbacks=[
            RegularCheckpoint(
                model=vae_model, 
                period=5000, 
                base_dir="checkpoints/vae", 
                do_img=True, 
                do_q=False
            ),
        ], 
        logger=vae_logger
    )

    while True:
        # Load checkpoint if it exists 
        if (os.path.exists("checkpoints/vae/model.ckpt")):
            vae_trainer.fit(vae_model, train_loader, val_loader, ckpt_path="checkpoints/vae/model.ckpt")
        else:
            vae_trainer.fit(vae_model, train_loader, val_loader)

def train():
    args = sys.argv[1:]
    if len(args) == 0:
        train_denoiser()
        print("Please specify a model to train.")
        return
    else:
        if args[0] == "denoiser":
            train_denoiser()
        elif args[0] == "vae":
            train_vae()
        else:
            print("Invalid model specified.")

if __name__ == '__main__':
    train()