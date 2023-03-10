# Data Loader for Unsplash Lite Dataset

import csv
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.OldCringeBERT import OldCringeBERTWrapper
from utils import *

class UnsplashLiteDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_dim=256):    
        self.image_paths = []
        self.image_captions = []

        self.im_dimension = img_dim

        bertWrapper = OldCringeBERTWrapper()

        # Get max length
        self.text_max = 512

        # Open the CSV file and read the image path from it
        with open(root_dir + '/manifest.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                image_path = root_dir + '/' + row[0]
                image_caption = row[1]
                image_caption = torch.tensor(bertWrapper.bert_tokenizer.encode(image_caption)).unsqueeze(0)
        
                #if (image_caption.size()[1] > self.text_max):
                #    self.text_max = image_caption.size()[1]
                if (image_caption.size()[1] >= self.text_max):
                    image_caption = image_caption[:, :self.text_max]
                else:
                    image_caption = torch.nn.functional.pad(image_caption, (0, self.text_max - image_caption.size()[1]), 'constant', 0)

                image_caption = image_caption.squeeze(0)

                self.image_paths.append(image_path)
                self.image_captions.append(image_caption)


    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        if (not os.path.exists(path)):
            return None, None
        else:
            x = Image.open(path)
            x = x.resize((self.im_dimension, self.im_dimension))
            x = np.array(x)
            if x.shape != (self.im_dimension, self.im_dimension, 3):
                print(f"Warning: image shape is not ({self.im_dimension}, {self.im_dimension}, 3). Skipping")
                print(x.shape)
                return None, None

            x = convert_to_tensor(x)
            x = x.squeeze(0)
            if x.shape != (3, self.im_dimension, self.im_dimension):
                print(f"Warning: image shape is not (3, {self.im_dimension}, {self.im_dimension}). Skipping")
                print(x.shape)
                return None, None

        q = self.image_captions[idx]
        return x, q