import open_clip
import pytorch_lightning as pl
import torch

from torch.nn import functional as F


class CringeCLIPModel(pl.LightningModule):
    """
        CringeCLIP
    """

    def __init__(self, model_type="RN50", hparams=None, has_cross_attention=False, img_dim=512):
        super().__init__()

        self.clip_module, _, self.preprocess = open_clip.create_model_and_transforms("RN50") # type: ignore
        self.tokenizer = open_clip.get_tokenizer("RN50") # type: ignore

    def forward(self, text = None, image = None):
        if (text is None) and (image is None):
            raise ValueError("Must provide either text or image")
        
        elif image is not None:
            x = self.clip_module.encode_image(image) # type: ignore
            x /= x.norm(dim=-1, keepdim=True)
            return x

        elif text is not None:
            x = self.clip_module.encode_text(text) # type: ignore
            x /= x.norm(dim=-1, keepdim=True)
            return x


    def configure_optimizers(self):
        """
            configure_optimizers

            This is the optimizer for the model.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-5)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        text, output = train_batch
        y = self.forward(text=text)
        loss = F.mse_loss(y, output) # type: ignore

        return loss
    
    def validation_step(self, train_batch, batch_idx):
        text, output = train_batch
        y = self.forward(text=text)
        loss = F.mse_loss(y, output) # type: ignore

        return loss