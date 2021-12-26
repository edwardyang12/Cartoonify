from flask import Flask, jsonify, request, render_template, Response
import torch
from PIL import Image
import io
import numpy as np
import os
import cv2

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator

size = 256
path = './unparallel.pth' # generator model is saved here

netG = Generator()

checkpoint = torch.load(path, map_location=torch.device('cpu'))
netG.load_state_dict(checkpoint['state_dict'])
print('loaded successfully')
netG.eval()

camera = cv2.VideoCapture(0)
def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


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

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method=='GET':
        return render_template('index.html', generated = '0')

    if request.method == 'POST':
        try:
            # we will get the file from the request
            file = request.files['file']
            # convert that to bytes
            img_bytes = file.read()
            picture = generate(image_bytes=img_bytes)
            return render_template('index.html', generated = str(picture))

        except:
            return render_template('index.html', generated = '2')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
