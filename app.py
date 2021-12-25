from flask import Flask, jsonify, request
import torch
from PIL import Image
import io
import numpy as np


# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator

size = 256
path = './unparallel.pth' # generator model is saved here

netG = Generator()

checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG.load_state_dict(checkpoint['state_dict'])
print('loaded successfully')
netG.eval()

def transform_image(image_bytes):
    MEAN = 255 * np.array([0.5, 0.5, 0.5])
    STD = 255 * np.array([0.5, 0.5, 0.5])

    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((size,size), resample=Image.BICUBIC)
    image = np.array(image).transpose(-1,0,1)
    image = (image - MEAN[:, None, None]) / STD[:, None, None]
    image = torch.tensor(image)
    return image.unsqueeze(0).type(torch.float32)

def generate(image_bytes):
    tensor = transform_image(image_bytes)
    outputs = netG.forward(tensor)
    outputs = outputs.detach().numpy().tolist()
    return outputs

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # we will get the file from the request
        file = request.files['file']
        # convert that to bytes
        img_bytes = file.read()
        picture = generate(image_bytes=img_bytes)

        return jsonify({'temp':picture})

if __name__ == '__main__':
    app.run()
