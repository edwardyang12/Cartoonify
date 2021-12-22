import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from nets.generator import OrigGenerator as Generator

load = True
size = 128
path = 'carolGAN.pth' # generator model is saved here

netG = Generator()
if(load):
    checkpoint = torch.load(path)
    netG.load_state_dict(checkpoint['state_dict'])
    print('loaded successfully')

fixed_noise = torch.randn(64, 3, 1, 1) # distribution sampler

with torch.no_grad():
    fake = netG(fixed_noise).numpy()

img = ((fake[0]*0.5)+0.5)*255.
img = img.transpose(1,2,0)
img = Image.fromarray(img.astype(np.uint8),'RGB')
img.save('newpic.png')
