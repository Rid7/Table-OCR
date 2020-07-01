# -*- coding: utf-8 -*-
# @Time    : 2019/8/24 12:06
# @Author  : zhoujun
from PSE.config.config import GPU_ID
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
import torch
from torchvision import transforms
import cv2
import math
from math import *
import numpy as np
import time
from PIL import Image

from PSE.models.model import PSENet
from PSE.pse import decode as pse_decode


def rotate(img, pt1, pt2, pt3, pt4):
    # print(pt1, pt2, pt3, pt4)
    widthRect = math.sqrt((pt4[0] - pt1[0]) ** 2 + (pt4[1] - pt1[1]) ** 2)  # 矩形框的宽度
    angle = acos((pt4[0] - pt1[0]) / widthRect) * (180 / math.pi)  # 矩形框旋转角度
    # print(angle)

    if pt4[1] > pt1[1]:
        angle = angle
        # print("顺时针旋转")
    else:
        # print("逆时针旋转")
        angle = -angle

    height = img.shape[0]  # 原始图像高度
    width = img.shape[1]  # 原始图像宽度
    rotateMat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)  # 按angle角度旋转图像
    heightNew = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
    widthNew = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

    rotateMat[0, 2] += (widthNew - width) / 2
    rotateMat[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(img, rotateMat, (widthNew, heightNew + 50), borderValue=(255, 255, 255))
    # cv2.imshow('rotateImg2',  imgRotation)
    # cv2.waitKey(0)

    # 旋转后图像的四点坐标
    [[pt1[0]], [pt1[1]]] = np.dot(rotateMat, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(rotateMat, np.array([[pt3[0]], [pt3[1]], [1]]))
    [[pt2[0]], [pt2[1]]] = np.dot(rotateMat, np.array([[pt2[0]], [pt2[1]], [1]]))
    [[pt4[0]], [pt4[1]]] = np.dot(rotateMat, np.array([[pt4[0]], [pt4[1]], [1]]))

    # 处理反转的情况
    if pt2[1] > pt4[1]:
        pt2[1], pt4[1] = pt4[1], pt2[1]
    if pt1[0] > pt3[0]:
        pt1[0], pt3[0] = pt3[0], pt1[0]

    imgOut = imgRotation[int(pt2[1]):int(pt4[1]), int(pt1[0]):int(pt3[0])]
    # cv2.imshow("imgOut", imgOut)  # 裁减得到的旋转矩形框
    # cv2.waitKey(0)
    return imgOut


class PSE_model:
    def __init__(self, model_path, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id
        net = PSENet(backbone='resnet152', pretrained=False)
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('text detection running on device:', self.device)

        if gpu_id is not None and isinstance(gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:{}".format(gpu_id))
        else:
            self.device = torch.device("cpu")

        self.net = torch.load(model_path, map_location=self.device)['state_dict']
        if net is not None:
            # 如果网络计算图和参数是分开保存的，就执行参数加载
            net = net.to(self.device)
            try:
                sk = {}
                for k in self.net:
                    sk[k[7:]] = self.net[k]
                net.load_state_dict(sk)
            except:
                net.load_state_dict(self.net)
            self.net = net
        print('device:', self.device)
        self.net.eval()


    def predict(self, img, scale_w, scale_h, ori_img):

        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(self.device)

        with torch.no_grad():
            # torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.net(tensor)[0]
            # torch.cuda.synchronize(self.device)
            preds, boxes_list = pse_decode(preds, scale=1, threshold=0.7311)
            if not boxes_list.any():
                return ''
            # print('boxes_list', boxes_list)
            # scale = (preds.shape[1] / w, preds.shape[0] / h)
            # print(33333333333333, scale)
            # if len(boxes_list):
            #     boxes_list = boxes_list / scale
            t = time.time() - start
            print('PANet inference time: {:.2f}s'.format(t))

        images = []
        boxes_list = boxes_list.astype('int32')
        boxes_list = boxes_list.reshape(-1, 8)
        # ori_img = np.array(ori_img)
        img_copy = ori_img.copy()
        for index, i in enumerate(boxes_list):
            i = [int(i[0] * scale_w), int(i[1] * scale_h), int(i[2] * scale_w), int(i[3] * scale_h),
                 int(i[4] * scale_w), int(i[5] * scale_h), int(i[6] * scale_w), int(i[7] * scale_h)]
            i = [i if i > 0 else 0 for i in i]
            if abs(i[1] - i[7]) < abs(i[0] - i[6]):
                i = [i[2], i[3], i[4], i[5], i[0], i[1], i[6], i[7]]
                x0, y0 = min(i[0], i[2], i[4], i[6]), min(i[1], i[3], i[5], i[7])
                new_im = ori_img[min(i[1], i[3], i[5], i[7]):max(i[1], i[3], i[5], i[7]),
                         min(i[0], i[2], i[4], i[6]):max(i[0], i[2], i[4], i[6])]
                nei_i = [ii - x0 if index in [0, 2, 4, 6] else ii - y0 for index, ii in enumerate(i)]
                try:
                    crop_img = rotate(new_im, [nei_i[0], nei_i[1]], [nei_i[4], nei_i[5]], [nei_i[6], nei_i[7]],
                                      [nei_i[2], nei_i[3]])
                except Exception as e:
                    print(1111111111, e)
                    print(i)
                    # cv2.imshow('rotateImg2', new_im)
                    continue
            else:
                i = [i[0], i[1], i[2], i[3], i[6], i[7], i[4], i[5]]
                x0, y0 = min(i[0], i[2], i[4], i[6]), min(i[1], i[3], i[5], i[7])
                new_im = ori_img[min(i[1], i[3], i[5], i[7]):max(i[1], i[3], i[5], i[7]),
                         min(i[0], i[2], i[4], i[6]):max(i[0], i[2], i[4], i[6])]
                nei_i = [ii - x0 if index in [0, 2, 4, 6] else ii - y0 for index, ii in enumerate(i)]
                try:
                    crop_img = rotate(new_im, [nei_i[0], nei_i[1]], [nei_i[4], nei_i[5]], [nei_i[6], nei_i[7]],
                                      [nei_i[2], nei_i[3]])
                except Exception as e:
                    print(22222222222, e)
                    continue
            cv2.line(img_copy, (i[0], i[1]), (i[2], i[3]), (255, 255, 0), 2)
            cv2.line(img_copy, (i[0], i[1]), (i[4], i[5]), (0, 255, 0), 2)
            cv2.line(img_copy, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
            cv2.line(img_copy, (i[4], i[5]), (i[6], i[7]), (255, 0, 0), 2)
            # Image.fromarray(crop_img).save('fast/{}.jpg'.format(index))
            # _, crop_img = cv2.imencode('.jpg', crop_img)
            # crop_img = base64.b64encode(crop_img.tostring())
            images.append([i, crop_img])

        # Image.fromarray(img_copy).save('1111.jpg')
        return images


if __name__ == '__main__':
    from PIL import ImageEnhance

    MAX_LEN = 2280

    text_detection_model = '/home/riddance/PycharmProjects/Table-OCR/pan/checkpoints/model_best.pth'
    text_detection_device = None
    text_detection = PSE_model(model_path=text_detection_model, gpu_id=text_detection_device)

