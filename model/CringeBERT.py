import pytorch_lightning as pl
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.utils.generic import ModelOutput
from transformers.modeling_outputs import BaseModelOutput


from model.cringe.unet import UNet

class CringeBERTWrapper:
    """
        BERT Wrapper

        This is a wrapper for the BERT model. Ideally would be trained from the same
        dataset as the LDM model, but for now we just use the pretrained BERT model.
    """

    def loadModel(self, cpu):
        self.bert_model = BertModel.from_pretrained(
            'bert-base-uncased')  # type: ignore
        if torch.cuda.is_available() & (not cpu):
            self.bert_model = self.bert_model.cuda()  # type: ignore
        self.bert_tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased')  # type: ignore

    def __init__(self, cpu=False):
        self.loadModel(cpu)
        pass

    def model_output(self, input_ids: torch.Tensor):
        with torch.no_grad():
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
            output = self.bert_model(input_ids)  # type: ignore
            q = output.last_hidden_state
            return q.unsqueeze(0)

    def inference(self, query):
        with torch.no_grad():
            # Encode the text using BERT
            input_ids: Tensor = torch.tensor(self.bert_tokenizer.encode(query)) \
                .unsqueeze(0)  # Add batch dimension
            # Normalise so that all values are between 0 and 1
            input_ids = (input_ids + 1) / 2
            return self.model_output(input_ids)