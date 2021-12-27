from flask import Flask, jsonify, request, render_template, Response, send_file
import torch
from PIL import Image
import io
import numpy as np
import os
import cv2
import requests
import base64

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
    outputs = outputs.detach().numpy()[0]
    return outputs

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    res = "data:image/png;base64,"
    if request.method=='GET':
        picture = np.zeros((32,32,3))
        picture.fill(255)
        ret, buffer = cv2.imencode('.jpg', picture)
        frame = buffer.tobytes()
        res+= str(base64.b64encode(frame))[2:]
        res = res[:len(res)-1]
        return render_template('index.html', result = 'initial', generated = res)

    if request.method == 'POST':
        try:
            # we will get the file from the request
            file = request.files['file']
            resp = requests.post("http://localhost:5000/gen_pic",
                                 files={"file": file})
            res+= str(resp.content)[2:]
            res = res[:len(res)-1]
            return render_template('index.html', result = 'success', generated = res)

        except:
            picture = np.zeros((32,32,3))
            picture.fill(255)
            ret, buffer = cv2.imencode('.jpg', picture)
            frame = buffer.tobytes()
            res+= str(base64.b64encode(frame))[2:]
            res = res[:len(res)-1]
            return render_template('index.html', result = 'failure', generated = res)

@app.route('/gen_pic', methods=['POST'])
def gen_pic():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        picture = generate(img_bytes).transpose(1,2,0)
        picture = np.uint8(((picture*0.5)+0.5)*255.)
        ret, buffer = cv2.imencode('.jpg', picture)
        frame = buffer.tobytes()
        res = base64.b64encode(frame)
        return res

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
