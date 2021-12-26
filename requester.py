import requests
from PIL import Image
import numpy as np

resp = requests.post("http://localhost:5000/predict",
                     files={"file": open('00000.png','rb')})

img = resp.json()['temp'] # simple test to make sure flask was able to send and receive a simple image 
img = np.array(img)
img = ((img[0]*0.5)+0.5)*255.
img = img.transpose(1,2,0)
img = Image.fromarray(img.astype(np.uint8),'RGB')
img.save('fake.png')
