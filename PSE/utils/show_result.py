import os
import cv2

gt_path = r'/home/rid/PythonProjects/PSENet.pytorch/output/output/'
img_path = r'/home/rid/Data/text_detection/validation/img/'
for img in os.listdir(img_path):
    img_copy = cv2.imread(os.path.join(img_path, img))
    with open(gt_path + 'res_' + os.path.splitext(img)[0] + '.txt', 'r') as f:
        for line in f:
            result = line.split(',')[:8]
            result = list(map(int, result))
            color = (255, 0, 0)
            thickness = 2
            cv2.line(img_copy, (result[0], result[1]), (result[2], result[3]), color, thickness)
            cv2.line(img_copy, (result[2], result[3]), (result[4], result[5]), color, thickness)
            cv2.line(img_copy, (result[4], result[5]), (result[6], result[7]), color, thickness)
            cv2.line(img_copy, (result[6], result[7]), (result[0], result[1]), color, thickness)
    cv2.imwrite('/home/rid/PythonProjects/PSE/output/{}'.format(img), img_copy)