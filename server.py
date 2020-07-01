import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import warnings
warnings.filterwarnings("ignore")
from simplified_unet import *
import numpy as np
import cv2
import base64
from PIL import Image, ImageEnhance
from sanic import Sanic, response


model = load_model('./model/model_17_0.9887387033492799.hdf5')
def unet_table(img):
    MAX_LEN = 1200
    img = Image.fromarray(img).convert('RGB')
    img = ImageEnhance.Contrast(img).enhance(3)
    # img.show()
    img = np.array(img)
    img_shape = img.shape[:2]
    print('Origin size:', img_shape)
    # image_size = (img_shape[1], img_shape[0])
    h, w = img_shape[1], img_shape[0]
    if (w < h):
        if (h > MAX_LEN):
            scale = 1.0 * MAX_LEN / h
            w = w * scale
            h = MAX_LEN
    elif (h <= w):
        if (w > MAX_LEN):
            scale = 1.0 * MAX_LEN / w
            h = scale * h
            w = MAX_LEN

    w = int(w // 16 * 16)
    h = int(h // 16 * 16)

    img_standard = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
    print('Scaled size:', img_standard.shape)
    img_new = np.expand_dims(img_standard/255., axis=0)

    unet_result = model.predict(img_new)
    unet_result = np.squeeze(unet_result, axis=-1)
    unet_result = np.squeeze(unet_result, axis=0)

    output_img = np.zeros_like(img_standard)
    output_img[unet_result < 0.3] = 255
    output_img = output_img.astype(np.uint8)

    # output_img = cv2.resize(output_img, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    print('Result size:', output_img.shape)

    output_img = cv2.imencode('.jpg', output_img)[1].tostring()
    output_img = base64.b64encode(output_img).decode()
    return output_img


# img = cv2.imread('test/00清研资本工商内档_55.jpg')
# img = cv2.imencode('.jpg', img)[1].tostring()
# img = base64.b64encode(img)
# img = base64.b64decode(img)
# img = np.fromstring(img, np.uint8)
# img = cv2.imdecode(img, cv2.IMREAD_COLOR)
# result = unet_table(img)

app = Sanic(__name__)

# app.config.update(DEBUG=True)
@app.route('/table_line', methods=['POST'])
def table_line(request):
    try:
        print('Getting image...')
        request_result = request.form.get("img")
        print('Table line predicting...', type(request_result))
        img_byte = base64.b64decode(request_result)
        img_np_arr = np.fromstring(img_byte, np.uint8)
        image = cv2.imdecode(img_np_arr, 1)
        output_img = unet_table(image)
        print('Table line result...', output_img)
        return response.json(output_img)
    except Exception as e:
        return response.json(e)


if __name__ == '__main__':
    app.config.KEEP_ALIVE = False
    app.config.REQUEST_TIMEOUT = 900
    app.config.RESPONSE_TIMEOUT = 900
    app.run(host='0.0.0.0', port=8282)