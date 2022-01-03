from flask import Flask, jsonify, request, render_template, Response, send_file
import torch
from PIL import Image
from threading import Thread
import io
import numpy as np
import os
import cv2
import requests
import base64

# from nets.generator import MiniUnet as Generator
from nets.newgenerator import ResnetGenerator as Generator
from face import face_detect

sizes = [128,256] # generator sizes
path = "./generators/" # generator model is saved here

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

generators = {}
for size in sizes:
    netG = Generator()
    checkpoint = torch.load(path+"gen"+str(size)+".pth", map_location=torch.device('cpu'))
    netG.load_state_dict(checkpoint['state_dict'])
    netG.eval()
    generators[size] = netG

print('loaded available networks successfully')

camera = cv2.VideoCapture(0)

global capture, modify
capture = False
modify = True

def take_pic():
    global capture, modify
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = np.array(frame)
            frame, faces = face_detect(frame, face_cascade)
            if modify: # if we want to generate pictures
                for face in faces:
                    x, y, w, h = face
                    crop_face = frame[y:y+h, x:x+w]
                    crop_face = generate(crop_face, (w+h)/2).transpose(1,2,0)
                    crop_face = np.uint8(((crop_face*0.5)+0.5)*255.)
                    crop_face = Image.fromarray(crop_face).resize((w,h), resample=Image.BICUBIC)
                    frame = exclude_bg(np.array(crop_face), frame, face)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
            while(capture):
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

def exclude_bg(generated, original, loc):
    x, y, w, h = loc
    mask = (generated>250.) # if white use original, else use generated
    mask = np.tile(np.expand_dims(np.all(mask, axis=2),2),3)
    original[y:y+h, x:x+w] = original[y:y+h, x:x+w] * mask.astype(int) + generated * (np.logical_not(mask)).astype(int)
    return original

def transform_image(image_bytes,size):
    MEAN = 255 * np.array([0.5, 0.5, 0.5])
    STD = 255 * np.array([0.5, 0.5, 0.5])
    image = Image.fromarray(np.uint8(image_bytes)).convert('RGB')
    image = image.resize((size,size), resample=Image.BICUBIC)
    image = np.array(image).transpose(-1,0,1)
    image = (image - MEAN[:, None, None]) / STD[:, None, None]
    image = torch.tensor(image)
    return image.unsqueeze(0).type(torch.float32)

def generate(image_bytes, size):
    val = 10000
    min = 0
    for i in sizes:
        if abs(size-i)<val:
            val = abs(size-i)
            min = i
    tensor = transform_image(image_bytes, min)
    outputs = generators[min].forward(tensor)
    outputs = outputs.detach().numpy()[0]
    return outputs

def standardPic(res):
    picture = np.zeros((32,32,3))
    picture.fill(255)
    ret, buffer = cv2.imencode('.jpg', picture)
    frame = buffer.tobytes()
    res+= str(base64.b64encode(frame))[2:]
    res = res[:len(res)-1]
    return res


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    res = "data:image/png;base64,"
    if request.method=='GET':
        return render_template('index.html', result = 'initial', original = standardPic(res), generated = standardPic(res))

    if request.method == 'POST':
        try:
            # we will get the file from the request
            file = request.files['file']
            resp = requests.post("http://localhost:5000/gen_pic",
                                 files={"file": file})
            res+= str(resp.content)[2:]
            res = res[:len(res)-1]

            file.seek(0)
            original = file.read()
            image = Image.open(io.BytesIO(original)).convert('RGB')

            image = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
            image = np.uint8(image)
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            file = "data:image/png;base64," + str(base64.b64encode(frame))[2:]
            return render_template('index.html', result = 'success',  original=file[:len(file)-1], generated = res)

        except:
            return render_template('index.html', result = 'failure', original = standardPic(res), generated = standardPic(res))


# sometimes crashes if u input images, i'm not 100% sure why
# I tried threading the function that prolly causes the issue but it still doesnt work
# at least now a crash doesnt bring down the whole app
@app.route('/gen_pic', methods=['POST'])
def gen_pic():
    if request.method == 'POST':
        global input_i, faces
        faces = []
        file = request.files['file']
        img_bytes = file.read()
        input_i = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        input_i = cv2.cvtColor(np.float32(input_i), cv2.COLOR_BGR2RGB)
        input_i = np.uint8(input_i)
        x = Thread(target=face_detect, args=[input_i, face_cascade, faces,])
        x.start()
        x.join()

        # image, faces = face_detect(image, face_cascade)
        for face in faces:
            x, y, w, h = face
            crop_face = input_i[y:y+h, x:x+w]
            crop_face = generate(crop_face, (w+h)/2).transpose(1,2,0)
            crop_face = np.uint8(((crop_face*0.5)+0.5)*255.)
            crop_face = Image.fromarray(crop_face).resize((w,h), resample=Image.BICUBIC)
            frame = exclude_bg(np.array(crop_face), input_i, face)
        ret, buffer = cv2.imencode('.jpg', input_i)
        frame = buffer.tobytes()
        res = base64.b64encode(frame)
        return res

@app.route('/video_feed')
def video_feed():
    return Response(take_pic(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests', methods=['GET', 'POST'])
def tasks():
    res = "data:image/png;base64,"
    global capture, modify
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            capture = not capture
        if request.form.get('click') == 'Modify':
            modify = not modify
    elif request.method == 'GET':
        pass
    return render_template('index.html', result = 'initial', original = standardPic(res), generated = standardPic(res))

if __name__ == '__main__':
    app.run(debug=True,port=os.getenv('PORT',5000))
