import requests
import base64
import cv2
import numpy as np
from PIL import Image
import time

url = 'http://127.0.0.1:8282/table_line'

image = cv2.imread(r"./test/test_3.jpg")
image_b64 = base64.b64encode(cv2.imencode('.jpg', image)[1])
start = time.time()
r = requests.post(url, data={"img": image_b64})

print(r.content.decode("utf-8"))
img_byte = base64.b64decode(r.content.decode("utf-8"))
img_np_arr = np.fromstring(img_byte, np.uint8)
image = cv2.imdecode(img_np_arr, cv2.IMREAD_COLOR)
end = time.time()
print(end-start)
Image.fromarray(image).save('./predict/test_3.jpg')