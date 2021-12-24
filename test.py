import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import itertools
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import sys

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator
from custom_dataset import TestDataset

size = 256

datapath = "./data/list_test_faces.csv"
savedir = sys.argv[1] # "/edward-slow-vol/cycleGAN/test/"

dataset = TestDataset(datapath,size)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

load = True
size = 128
path = '/edward-slow-vol/cycleGAN/cycle128/cycleGAN49.pth' # generator model is saved here

netG = Generator().to(device)
netG = nn.DataParallel(netG, list(range(1)))
if(load):
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['state_dict'])
    print('loaded successfully')

for i, data in enumerate(dataloader):
    simdata, simpath, = data
    b_size,channels,h,w = simdata.shape

    with torch.no_grad():
        fake = netG(simdata).detach().cpu().numpy()
    img = ((fake[0]*0.5)+0.5)*255.
    img = img.transpose(1,2,0)
    img = Image.fromarray(img.astype(np.uint8),'RGB')
    img.save(savedir+'fake'+str(i)+'.png')
    print(simpath)
