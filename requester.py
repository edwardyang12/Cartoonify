import requests
from PIL import Image
import numpy as np

resp = requests.post("http://localhost:5000/gen_pic",
                     files={"file": open('00000.png','rb')})
print(resp.content)

# simple test to make sure flask was able to send and receive a simple image, deprecated now
# img = resp.json()
# img = np.array(img)
# img = ((img[0]*0.5)+0.5)*255.
# img = img.transpose(1,2,0)
# img = Image.fromarray(img.astype(np.uint8),'RGB')
# img.save('fake.png')
