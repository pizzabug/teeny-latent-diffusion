# Data Loader for Unsplash Lite Dataset

import csv
import numpy as np
import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from model.CringeLDM import CringeBERTWrapper

class UnsplashLiteDataset(Dataset):
    def __init__(self, root_dir, transform=None):    
        self.image_paths = []
        self.image_captions = []

        bertWrapper = CringeBERTWrapper()

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
            image = None
        else:
            image = Image.open(path)
            image = image.resize((256, 256))
            image = np.array(image)
            image = image / 255.0
            image = image.transpose(2, 0, 1)
            image = torch.tensor(image, dtype=torch.float32)
            
            if image.shape != (3, 256, 256):
                print("Warning: image shape is not (3, 256, 256). Skipping")
                print(image.shape)
                image = None

        q = self.image_captions[idx]
        return image, q