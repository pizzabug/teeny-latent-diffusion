import torch
import torch.nn as nn
import transformers
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
        self.bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

    def __init__(self):
        self.loadModel()
        pass

    def inference (self, query):
        with torch.no_grad():
            # Encode the text using BERT
            input_ids = torch.tensor(self.bert_tokenizer.encode(query)).unsqueeze(0)  # Add batch dimension
            modelOutput : BaseModelOutput = self.bert_model(input_ids)
            return  modelOutput.last_hidden_state # Take the output for the first input in the batch

"""
    LDM Model

    This is the definition of the LDM model.
"""
class CringeLDM(nn.Module):
    def __init__(self):
        super().__init__()
        self.UNet = UNetWithCrossAttention(256,256,768)

    def forward(self, x, query):
        # Do Denoising UNet
        x = self.UNet(x, query)
        return x

"""
    LDM Wrapper

    This is a wrapper for the LDM model.
"""
class CringeLDMWrapper:
    def __init__(self, bert: CringeBERTWrapper = None, ldm = None):
        if bert is None:
            self.bert = CringeBERTWrapper()
            self.bert.loadModel()
        else:
            self.bert = bert
        
        if ldm is None:
            self.createModel()
        else:
            self.ldm = ldm

    #def __init__(self, ldm, bert: CringeBERTWrapper):
    #    self.ldm = ldm
    #    self.bert = bert
    
    def createModel (self):
        self.ldm = CringeLDM()
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.ldm.parameters(), lr=1e-3)

    def inference (self, steps = 20, query = "", img = None):
        # Load the image
        if img is None:
            # Generate noise
            img = torch.randn(1, 3, 256, 256)
        
        # Get the BERT output
        query = self.bert.inference(query)

        # We denoise for multiple steps
        for i in range(steps):
            img = self.ldm(img, query)
        
        return img
        
