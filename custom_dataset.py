import os
from torch.utils.data import Dataset
import torch
from PIL import Image, ImageFile
import numpy as np
import random
import pandas as pd

ImageFile.LOAD_TRUNCATED_IMAGES = True


def normalize(image):
    MEAN = 255 * torch.tensor([0.5, 0.5, 0.5])
    STD = 255 * torch.tensor([0.5, 0.5, 0.5])

    x = torch.from_numpy(np.array(image))
    x = x.type(torch.float32)
    x = x.permute(-1,0,1)
    x = (x - MEAN[:, None, None]) / STD[:, None, None]
    return x

class CustomDataset(Dataset):
    def __init__(self, facespath, carpath, size):
        self.faces = pd.read_csv(facespath)
        self.cartoon = pd.read_csv(carpath)
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
        temp = random.randrange(0,len(self.cartoon))
        cartoon, rName = self.load_image(self.cartoon, temp)
        cartoon = normalize(cartoon)
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
