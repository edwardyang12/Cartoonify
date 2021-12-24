import os
from torch.utils.data import Dataset
import torchvision.transforms as Transforms
import torch
from PIL import Image, ImageFile
import numpy as np
import random
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True

class CustomDataset(Dataset):
    def __init__(self, facespath, carpath, size):
        self.faces = pd.read_csv(facespath)
        self.cartoon = pd.read_csv(carpath)
        self.size = size

    def data_aug(self):
        transform_list = [
            Transforms.ToTensor(),
            Transforms.Normalize(
                mean=255.*np.array([0.485, 0.456, 0.406]),
                std=255.*np.array([0.229, 0.224, 0.225]),
            ),
            Transforms.RandomHorizontalFlip(),
            Transforms.ColorJitter(brightness=.1, contrast=.1,saturation=.1),
        ]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def load_image(self, filename, index, cartoon=False):
        name = filename.values[index][1]
        temp = Image.open(name).convert('RGB')
        if cartoon: # because the cartoon face has a lot of white space
            w, h = temp.size
            w_sub = w*.1
            h_sub = h*.1
            temp = temp.crop((w_sub,h_sub,w-w_sub,h-h_sub))
        temp = temp.resize((self.size,self.size), resample=Image.BICUBIC)
        return np.array(temp).astype(np.float32), name
    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        transform = self.data_aug()
        faces, name = self.load_image(self.faces, index)
        faces = transform(faces)
        temp = random.randrange(0,len(self.cartoon))
        cartoon, rName = self.load_image(self.cartoon, temp, cartoon=True)
        cartoon = transform(cartoon)
        return faces, name, cartoon, rName

class TestDataset(Dataset):
    def __init__(self, facespath, size):
        self.faces = pd.read_csv(facespath)
        self.size = size

    def load_image(self, filename, index):
        name = filename.values[index][1]
        temp = Image.open(name).convert('RGB')
        temp = temp.resize((self.size,self.size), resample=Image.BICUBIC)
        return np.array(temp).astype(np.float32), name

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        faces, name = self.load_image(self.faces, index)
        faces = normalize(faces)
        return faces, name
