import pytorch_lightning as pl
import torch
import torch.nn as nn
import transformers

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers import BertModel
from transformers.utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput

from model.cringe.unet import UNetWithCrossAttention

"""
    BERT Wrapper

    This is a wrapper for the BERT model. Ideally would be trained from the same
    dataset as the LDM model, but for now we just use the pretrained BERT model.
"""
class CringeBERTWrapper:
    def loadModel (self):
        self.bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
        if torch.cuda.is_available():
            self.bert_model = self.bert_model.cuda()
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self):
        self.loadModel()
        pass

    def model_output (self, input_ids : torch.Tensor) -> BaseModelOutput:
        with torch.no_grad():
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            output = self.bert_model(input_ids)
            return output.last_hidden_state

    def inference (self, query):
        with torch.no_grad():
            # Encode the text using BERT
            input_ids : Tensor = torch.tensor(self.bert_tokenizer.encode(query)).unsqueeze(0)  # Add batch dimension
            return self.model_output(input_ids)

"""
    LDM Model

    This is the definition of the LDM model.
"""
class CringeLDM(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # BERT Wrapper for the text encoding
        # This should be an integrated part of the model
        # in the future
        self.bertWrapper = CringeBERTWrapper()
        # Latent space
        self.UNet = UNetWithCrossAttention(256,256,768)
        # Image space decoder
        self.imageSpaceDecoder = nn.Sequential(
            nn.Conv2d(3, 6, 12, padding='same'),
            nn.ReLU(),
            nn.Conv2d(6, 3, 24, padding='same'),
            nn.Dropout(0.02),
            nn.ReLU(),
            nn.Conv2d(3, 3, 12, padding='same'),
        )

    """
        forward
        
        self: The model
        q: Query tensor from BERT
        x: Image tensor
        steps: Number of steps to denoise the image
    """
    def forward(self, q, x = None, steps = 20):        
        # Load the image
        if x is None:
            # Generate noise
            x = torch.randn(q.shape[0], 3, 256, 256)
        
        if torch.cuda.is_available():
            x = x.cuda()
            q = q.cuda()

        # We denoise for multiple steps
        for i in range(steps):
            # This is the latent space
            x = self.UNet(x, q)

            # Image space decoder
            x = self.imageSpaceDecoder(x)

        return x

    """
        configure_optimizers

        This is the optimizer for the model.
    """
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
    """
        training_step
    """
    def training_step(self, train_batch, batch_idx):
        # Grab batch
        y, q = train_batch

        # Cuda up if needed
        if torch.cuda.is_available():
            y = y.cuda()
            q = q.cuda()

        # Get q
        q = self.bertWrapper.model_output(q)
        # Forward pass
        y_hat = self.forward(q)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    """
        validation_step
    """
    def validation_step(self, val_batch, batch_idx):
        # Grab batch
        y, q = val_batch

        # Cuda up if needed
        if torch.cuda.is_available():
            y = y.cuda()
            q = q.cuda()

        # Get q
        q = self.bertWrapper.model_output(q)
        # Forward pass
        y_hat = self(q)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def forward_with_q (self, query, x = None, steps = 20):
        
        # Get the BERT output
        q = self.bertWrapper.inference(query)

        if torch.cuda.is_available():
            q = q.cuda()
            if (x != None):
                x = x.cuda()

        # Forward pass
        return self.forward(q, x, steps)