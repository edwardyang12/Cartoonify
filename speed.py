import time
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator

size = 256
trials = 100
path = './unparallel.pth' # generator model is saved here

netG = Generator()
print(netG)

checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG.load_state_dict(checkpoint['state_dict'])
print('loaded successfully')
netG.eval()

# original (double/ float 64)
tensor = torch.zeros(1,3,size,size)
# average_t = 0
# for i in range(trials):
#     start = time.time()
#     outputs = netG.forward(tensor)
#     average_t += time.time()-start
#
# average_t = average_t/trials
# print("Original Generator Inference Time")
# print(average_t)

# work in progress
netG.qconfig = torch.quantization.get_default_qconfig('qnnpack')

netG_fused = torch.quantization.fuse_modules(netG, [['conv', 'bn', 'relu']]) # doesn't work
netG_prepared = torch.quantization.prepare(netG)

netG_prepared(tensor)
model_int8 = torch.quantization.convert(netG_prepared)
output = model_int8(tensor) # need to add quant/ dequant to the model
