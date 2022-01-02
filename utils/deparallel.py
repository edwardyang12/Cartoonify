import torch
import torch.nn as nn
from collections import OrderedDict

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator

savedir = "/edward-slow-vol/cycleGAN/"

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

path = '/edward-slow-vol/cycleGAN/cycle128/cycleGAN49.pth' # generator model is saved here

netG = Generator().to(device)
netG = nn.DataParallel(netG, list(range(1)))
checkpoint = torch.load(path)
netG.load_state_dict(checkpoint['state_dict'])
print('loaded successfully')

new_state_dict = OrderedDict()
for k, v in (checkpoint['state_dict']).items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v

filename = savedir + 'unparallel.pth'
state = {'state_dict': new_state_dict}
torch.save(state, filename)
print('saved')
