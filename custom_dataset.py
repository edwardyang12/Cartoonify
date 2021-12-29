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
            Transforms.GaussianBlur(kernel_size=9, sigma=(0.1,2)),
            Transforms.ColorJitter(brightness=.2),
            Transforms.RandomHorizontalFlip(),
            Transforms.Normalize(
                mean=np.array([0.5, 0.5, 0.5]),
                std=np.array([0.5, 0.5, 0.5]),
            ),
        ]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def load_image(self, filename, index, cartoon=False):
        name = filename.values[index][1]
        temp = Image.open(name).convert('RGB')
        if cartoon: # because the cartoon face has a lot of white space
            w, h = temp.size
            w_sub = w*.15
            h_sub = h*.15
            temp = temp.crop((w_sub,h_sub,w-w_sub,h-h_sub))
        temp = temp.resize((self.size,self.size), resample=Image.BICUBIC)
        return temp, name

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        transform = self.data_aug()
        faces, name = self.load_image(self.faces, index)
        faces = transform(faces)

        temp = random.randrange(0,len(self.cartoon))
        cartoon, rName = self.load_image(self.cartoon, temp, cartoon=True)
        transform = self.data_aug()
        cartoon = transform(cartoon)
        return faces, name, cartoon, rName

class TestDataset(Dataset):
    def __init__(self, facespath, size):
        self.faces = pd.read_csv(facespath)
        self.size = size

    def data_aug(self):
        transform_list = [
            Transforms.ToTensor(),
            Transforms.Normalize(
                mean=np.array([0.5, 0.5, 0.5]),
                std=np.array([0.5, 0.5, 0.5]),
            ),
        ]
        custom_augmentation = Transforms.Compose(transform_list)
        return custom_augmentation

    def load_image(self, filename, index):
        name = filename.values[index][1]
        temp = Image.open(name).convert('RGB')
        temp = temp.resize((self.size,self.size), resample=Image.BICUBIC)
        return temp, name

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, index):
        transform = self.data_aug()
        faces, name = self.load_image(self.faces, index)
        faces = transform(faces)
        return faces, name
