import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import sys
import cv2

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator
from face import face_detect

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

size = 256

path = "./generators/" # generator model is saved here
target = "1.jpg"

netG = Generator()
checkpoint = torch.load(path+"gen"+str(size)+".pth", map_location=torch.device('cpu'))
netG.load_state_dict(checkpoint['state_dict'])
netG.eval()
print('loaded successfully')

def exclude_bg(generated, original, loc):
    x, y, w, h = loc
    mask = (generated>250.) # if white use original, else use generated
    mask = np.tile(np.expand_dims(np.all(mask, axis=2),2),3)
    original[y:y+h, x:x+w] = original[y:y+h, x:x+w] * mask.astype(int) + generated * (np.logical_not(mask)).astype(int)
    return original

def generate(image_bytes):
    MEAN = 255 * np.array([0.5, 0.5, 0.5])
    STD = 255 * np.array([0.5, 0.5, 0.5])
    image = Image.fromarray(np.uint8(image_bytes)).convert('RGB')
    image = image.resize((size,size), resample=Image.BICUBIC)
    image = np.array(image).transpose(-1,0,1)
    image = (image - MEAN[:, None, None]) / STD[:, None, None]
    image = torch.tensor(image).unsqueeze(0).type(torch.float32)
    outputs = netG.forward(image)
    outputs = outputs.detach().numpy()[0]
    return outputs

image = Image.open(target).convert('RGB')
frame = np.array(image)
frame, faces = face_detect(frame, face_cascade)
for i, face in enumerate(faces):
    x, y, w, h = face
    crop_face = frame[y:y+h, x:x+w]
    Image.fromarray(crop_face).save("crop" + str(i)+ ".jpg")
    crop_face = generate(crop_face).transpose(1,2,0)
    crop_face = np.uint8(((crop_face*0.5)+0.5)*255.)
    crop_face = Image.fromarray(crop_face).resize((w,h), resample=Image.BICUBIC)

    frame = exclude_bg(np.array(crop_face), frame, face)
outputs = Image.fromarray(np.uint8(frame)).convert('RGB')
outputs.save("output.jpg")
