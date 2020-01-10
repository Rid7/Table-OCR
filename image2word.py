import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import cv2
import numpy as np
# import torch

from PIL import Image
from docx import Document

from restore_table import restore_table
# from Rec_text import rec_txt
# from pan.predict import text_predict
# from crnn_torch.model1 import predict
# from ctpn.ctpn_blstm_test_full import text_predict
# from densent_ocr.model import predict
# from crnn_seq2seq_ocr.inference import attention
# from viterbi import calculate


def remove_line(src, FLAG, max_depth=10, gap_rate=0.05, epsilon=1e-6):
    """
    :param src: Source line image: threshold mode
    :param FLAG: 0 for removing horizontal line, 1 for removing vertical line
    :param max_depth: recurse max depth
    :param gap_rate: Rate of the longest line subtract the second longest line divide the longest line
    :param epsilon: Prevent from dividing zero error
    :return: After remove line result: threshold mode
    """
    if max_depth:
        if FLAG == 0:
            contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            horizontal_left = []
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(np.array(contours[i]))
                horizontal_left.append((x, y, w, h))

            if len(horizontal_left) >= 2:
                horizontal_left = sorted(horizontal_left, key=lambda i: i[0])
                horizontal_right = sorted(horizontal_left, key=lambda i: i[0] + i[2])

                if abs(horizontal_left[1][0] - horizontal_left[0][0]) / (horizontal_left[1][0] + epsilon) > gap_rate:
                    x, y, w, h = horizontal_left[0]
                    src[y:y+h, x:x+w] = 0

                elif abs(horizontal_right[-1][0] + horizontal_right[-1][2] - \
                        (horizontal_right[-2][0] + horizontal_right[-2][2])) / \
                        (horizontal_right[-1][0] + horizontal_right[-1][2] + epsilon) > gap_rate:
                    x, y, w, h = horizontal_right[-1]
                    src[y:y+h, x:x+w] = 0
                else:
                    return src
            else:
                return src
            max_depth = max_depth - 1
            return remove_line(src, FLAG=0, max_depth=max_depth)

        elif FLAG == 1:
            contours, _ = cv2.findContours(src, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            vertical_up = []
            for i in range(len(contours)):
                x, y, w, h = cv2.boundingRect(np.array(contours[i]))
                vertical_up.append((x, y, w, h))

            if len(vertical_up) >= 2:
                vertical_up = sorted(vertical_up, key=lambda i: i[1])
                vertical_down = sorted(vertical_up, key=lambda i: i[1] + i[3])

                if abs(vertical_up[1][1] - vertical_up[0][1]) / (vertical_up[1][1] + epsilon) > gap_rate:
                    x, y, w, h = vertical_up[0]
                    src[y:y+h, x:x+w] = 0

                elif abs(vertical_down[-1][1] + vertical_down[-1][3] - \
                         (vertical_down[-2][1] + vertical_down[-2][3])) /\
                        (vertical_down[-1][1] + vertical_down[-1][3] + epsilon) > gap_rate:

                    x, y, w, h = vertical_down[-1]
                    src[y:y+h, x:x+w] = 0
                else:
                    return src
            else:
                return src
            max_depth = max_depth - 1
            return remove_line(src, FLAG=1, max_depth=max_depth)
    else:
        return src


def get_roi(src, FLAG, mask):
    """
    :param src: Source image: RGB mode
    :param FLAG: 0 for the 1st time getting roi, 1 for the 2nd time
    :param mask: Line image: threshold mode
    :return: rois: rois[i][0]: region of interest in the source image: RGB mode,
    rois[i][1]: roi coordinates of source image
    """
    mark_src = src.copy()
    if FLAG == 0:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mark_src, contours, -1, (255, 0, 0), 2)
        Image.fromarray(mark_src).show()
    elif FLAG == 1:
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mark_src, contours, -1, (255, 0, 0), 2)

    contours_poly = [''] * len(contours)
    rois = []
    for i, j in zip(range(len(contours)), hierarchy[0]):
        # print(j)
        min_rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(min_rect)
        box = np.int0(box)
        cv2.drawContours(mark_src, [box], -1, (255, 0, 0), 3)

        if FLAG == 0 or j[3] == 0:
            area = cv2.contourArea(contours[i])
            min_rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(min_rect)
            box = np.int0(box)
            cv2.drawContours(mark_src, [box], 0, (255, 0, 0), 3)
            contours_poly[i] = cv2.approxPolyDP(contours[i], 10, True)

            min_rect_h, min_rect_w = min_rect[1][0], min_rect[1][1]
            x, y, w, h = cv2.boundingRect(np.array(contours_poly[i]))
            cv2.rectangle(mark_src, (x, y), (x + w, y + h), (0, 255, 0), 5)

            if FLAG == 0 and any([h < 15, w < 15, min_rect_h < 15, min_rect_w < 15]) or area < 100:
                continue

            ytt = src[y:y+h, x:x+w]
            # if FLAG == 1:
            #     mark_src[y:y+h, x:x+w] = 0

            rois.append([ytt, [x, y, w, h]])

        elif FLAG == 1 and j[3] > 0:
            contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)
            x, y, w, h = cv2.boundingRect(np.array(contours_poly[i]))
            cv2.rectangle(mark_src, (x, y), (x + w, y + h), (0, 0, 255), 2)

    Image.fromarray(mark_src).show()
    return rois


def rotate_degree(src, vertical):
    """
    :param src: Source image: RGB mode
    :param vertical: Vertical line image：threshold mode
    :return: if vertical image and Houghline detecting result exists: return the calculate degree,
    else return the default 0
    """
    mark_src = src.copy()
    if len(np.nonzero(vertical)) != 1:
        lines = cv2.HoughLinesP(vertical, 2, np.pi / 180, 150, minLineLength=50, maxLineGap=30)

        angles = []
        if len(np.nonzero(lines)) != 1:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if length / vertical.shape[0] > 0.2:
                    angle = np.arcsin((x2 - x1) / length)
                    angles.append(angle)
                    cv2.line(mark_src, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            return 0

        Image.fromarray(mark_src).show()
        mean_angle = np.mean(angles)
        degree = mean_angle * (180 / np.pi)
        print('rotate degree:', degree)
        return degree
    return 0


def warp_perspective(src, vertical, horizontal):
    """
    :param src: 1st roi in source image: RGB mode
    :param vertical: 1st roi in vertical line image: threshold mode
    :param horizontal: 1st roi in horizontal line image: threshold mode
    :return: After warp perspective src, vertical, horizontal image,
    and processed mark：False for not processed, True for processed
    """
    mark_src = src.copy()
    height = src.shape[0]
    width = src.shape[1]
    mask = vertical + horizontal
    
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dst_corner = np.array(((0, 0),
                           (width - 1, 0),
                           (width - 1, height - 1),
                           (0, height - 1)),
                           np.float32)

    approx = cv2.approxPolyDP(contours[0], 10, True)
    hull = cv2.convexHull(approx, clockwise=True)
    
    cv2.polylines(mark_src, [hull], True, (0, 0, 255), 2)
    Image.fromarray(mark_src).show()
    
    print(hull)
    if len(hull) != 4:
        processed = False
        return src, vertical, horizontal, processed

    src_corner = hull.reshape(4, 2)
    src_corner = np.float32(src_corner)
    center_x, center_y = np.mean(src_corner, axis=0)
    sort_key = np.select([np.logical_and(src_corner[:, 0] < center_x, src_corner[:, 1] < center_y),
                          np.logical_and(src_corner[:, 0] > center_x, src_corner[:, 1] < center_y),
                          np.logical_and(src_corner[:, 0] > center_x, src_corner[:, 1] > center_y),
                          np.logical_and(src_corner[:, 0] < center_x, src_corner[:, 1] > center_y)],
                          [0.0, 1.0, 2.0, 3.0])
    sort_key = np.expand_dims(sort_key, axis=0)
    temp = np.concatenate((src_corner, sort_key.T), axis=1)
    sorted_corner = temp[np.lexsort(temp.T)]
    sorted_corner = sorted_corner[:, 0:2]

    M, _ = cv2.findHomography(sorted_corner, dst_corner, cv2.RANSAC, 10)
    src = cv2.warpPerspective(src, M, (width, height))
    vertical = cv2.warpPerspective(vertical, M, (width, height))
    horizontal = cv2.warpPerspective(horizontal, M, (width, height))
    processed = True
    return src, vertical, horizontal, processed


def table_lines(src, horizontal=[], vertical=[]):
    """
    :param src: 1st roi in Source image: RGB mode
    :param horizontal: 1st roi in horzontal line image: threshold mode
    :param vertical: 1st roi in vertical line image: threshold mode
    :return:
    """
    # MAX_LEN = 1200
    # img = Image.open('test/{}'.format(file)).convert('RGB')
    # img = ImageEnhance.Contrast(img).enhance(3)
    # img.show()
    # img = np.array(img)
    # img_shape = img.shape[:2]
    # print('Origin size:', img_shape)
    # image_size = (img_shape[1], img_shape[0])
    # h, w = img_shape[1], img_shape[0]
    # if (w < h):
    #     if (h > MAX_LEN):
    #         resize_rate = 1.0 * MAX_LEN / h
    #         w = w * resize_rate
    #         h = MAX_LEN
    # elif (h <= w):
    #     if (w > MAX_LEN):
    #         resize_rate = 1.0 * MAX_LEN / w
    #         h = resize_rate * h
    #         w = MAX_LEN
    #
    # w = int(w // 16 * 16)
    # h = int(h // 16 * 16)
    # # h, w = 512, 512
    #
    # roi = cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
    # roi = roi[coor[1]:coor[1]+coor[3],coor[0]:coor[0]+coor[2],:]
    # # cv2.putText(roi, 'roi', (20, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 5)
    # # Image.fromarray(roi).show()
    # src_height, src_width = roi.shape[:2]
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # print(src_height)
    #
    # thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
    #
    # roi_horizontal = thresh
    # roi_vertical = thresh
    #
    # roi_horizontalsize = int(src_height / scale)
    # print(roi_horizontalsize)
    # roi_horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (roi_horizontalsize, 1))
    # roi_horizontal = cv2.erode(roi_horizontal, roi_horizontalStructure, (-1, -1))
    # roi_horizontal = cv2.dilate(roi_horizontal, roi_horizontalStructure, (-1, -1))
    # roi_horizontal = cv2.blur(roi_horizontal, (5, 5))
    #
    # roi_verticalsize = int(src_width / scale)
    # roi_verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, roi_verticalsize))
    # roi_vertical = cv2.erode(roi_vertical, roi_verticalStructure, (-1, -1))
    # roi_vertical = cv2.dilate(roi_vertical, roi_verticalStructure, (-1, -1))
    # roi_vertical = cv2.blur(roi_vertical, (5, 5))
    # mask = roi_vertical + roi_horizontal + vertical + horizontal

    src, vertical, horizontal, processed = warp_perspective(src, vertical, horizontal)
    if not processed:
        degree = rotate_degree(src, vertical)

        if abs(degree) < 10:
            horizontal = np.array(Image.fromarray(horizontal).rotate(degree))
            vertical = np.array(Image.fromarray(vertical).rotate(degree))
            src = np.array(Image.fromarray(src).rotate(degree))

    # 判断是否加线
    h_x, h_y, h_w, h_h = cv2.boundingRect(horizontal)
    v_x, v_y, v_w, v_h = cv2.boundingRect(vertical)

    OFFSET = 5
    cv2.line(vertical, (h_x + OFFSET, h_y + OFFSET), (h_x + OFFSET, h_y + h_h - OFFSET), 255, 3)
    cv2.line(vertical, (h_x + h_w - OFFSET, h_y + OFFSET), (h_x + h_w - OFFSET, h_y + h_h - OFFSET), 255, 3)
    cv2.line(horizontal, (v_x + OFFSET, v_y + OFFSET), (v_x + v_w - OFFSET, v_y), 255, 3)
    cv2.line(horizontal, (v_x + OFFSET, v_y + v_h - OFFSET), (v_x + v_w - OFFSET, v_y + v_h - OFFSET), 255, 3)

    mask = horizontal + vertical
    Image.fromarray(vertical).show()

    joints = cv2.bitwise_and(horizontal, vertical)
    joints = cv2.dilate(joints, None)

    if not joints.any():
        return False

    # contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # for cnt in contours:
    #     [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    #     lefty = int((-x * vy / vx) + y)
    #     righty = int(((src_width - x) * vy / vx) + y)
    #     cv2.line(src, (src_width - 1, righty), (0, lefty), (0, 255, 0), 2)
    #
    # Image.fromarray(src).show()
    
    cells = get_roi(src, 1, mask)

    Image.fromarray(mask).show()
    # cv2.imwrite('mask/' + file, src)
    joints_contours, _ = cv2.findContours(joints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Image.fromarray(joints).show()
    
    x_coor = []
    y_coor = []
    for i in joints_contours:
        x_coor.append(cv2.minEnclosingCircle(i)[0][0])
        y_coor.append(cv2.minEnclosingCircle(i)[0][1])

    x_coor = sorted(x_coor)
    y_coor = sorted(y_coor)
    
    xs = set()
    for index in range(len(x_coor) - 1):
        if abs(x_coor[index] - x_coor[index + 1]) < 12:
            x_coor[index + 1] = x_coor[index]
            xs.add(x_coor[index])

    ys = set()
    for index in range(len(y_coor) - 1):
        if abs(y_coor[index] - y_coor[index + 1]) < 12:
            y_coor[index + 1] = y_coor[index]
            ys.add(y_coor[index])

    print(xs, ys)
    print(len(cells))
    cols, rows, col_point, row_points = len(xs) - 1, len(ys) - 1, list(xs), list(ys)
    return cols, rows, col_point, row_points, cells


def draw_line(src, scale=30):
    if not src.data:
        print('Not picture!')
        return [], [], [], []

    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    erode_size = 3
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10)
    Image.fromarray(thresh).show()
    horizontal = thresh
    vertical = thresh

    horizontalsize = int(horizontal.shape[0] / scale)
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
    horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1))
    horizontal = cv2.blur(horizontal, (5, 5))

    verticalsize = int(vertical.shape[1] / scale)
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
    vertical = cv2.blur(vertical, (5, 5))

    Image.fromarray(vertical).show()
    Image.fromarray(horizontal).show()

    degree = rotate_degree(src, vertical)

    print('first stage degree:', degree)
    if abs(degree) < 1.5:
        horizontal = remove_line(horizontal, FLAG=0, gap_rate=0.05)
        vertical = remove_line(vertical, FLAG=1, gap_rate=0.05)

    Image.fromarray(vertical).show()
    Image.fromarray(horizontal).show()
    mask = horizontal + vertical
    # cv2.imwrite('mask/' + file, mask)
    
    joints = cv2.bitwise_and(horizontal, vertical)
    # if not joints.any():
    #     print('No joint！')
    #     return False

    # 判断交点数，小于2则不为表格
    tree_contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x_coor = []
    for i in tree_contours:
        x_coor.append(cv2.minEnclosingCircle(i)[0][0])

    x_coor = sorted(x_coor)
    xs = set()
    for index in range(len(x_coor) - 1):
        if abs(x_coor[index] - x_coor[index + 1]) < 5:
            x_coor[index + 1] = x_coor[index]
            xs.add(x_coor[index])
    # print('xs', len(xs))
    if len(xs) <= 2:
        print('Joints not satisfied！')
        return [], src, horizontal, vertical

    rois = get_roi(src, 0, mask)
    return rois, src, horizontal, vertical


def extract_table(image, file):
    # image = Image.fromarray(ori_image).copy()
    # image.thumbnail((600, 600), Image.ANTIALIAS)
    image = image.copy()
    print(image.shape)

    rois, src, horizontal, vertical = draw_line(image)

    if not rois:
        return []

    sort_table = sorted(rois, key=lambda i: i[1][3])
    tables = [sort_table[0]]

    for i in sort_table[1:]:
        count = 0
        for j in tables:
            if j[1][1] < i[1][1] + 10 and i[1][1] - 10 < j[1][1] + j[1][3]:
                continue
            else:
                count += 1
        if count == len(tables):
            tables.append(i)

    cells = []
    for i in tables:
        cols, rows, col_point, row_point, blocks = table_lines(src=i[0], horizontal=horizontal[i[1][1]:i[1][1]+
                                                               i[1][3],i[1][0]:i[1][0]+i[1][2]],\
                                                               vertical=vertical[i[1][1]:i[1][1]+i[1][3],i[1][0]:i[1][0]+i[1][2]])
        print('cols and rows', cols, rows)
        Image.fromarray(i[0]).show()
        if cols > 1 and rows > 1:
            cells.append([i, cols, rows, col_point, row_point, blocks])

    if cells:
        return cells
    else:
        return 'not table'


import traceback
from docx.oxml.shared import OxmlElement, qn


def set_cell_vertical_alignment(cell, align="center"):
    try:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        tcValign = OxmlElement('w:vAlign')
        tcValign.set(qn('w:val'), align)
        tcPr.append(tcValign)
        return True
    except:
        traceback.print_exc()
        return False


# extract_table(cv2.imread('table.jpg'))
def generate_table(cell, ori_img):
    # import pickle
    # pickle.dump(cell, open('table.pkl', 'wb'))
    pos, cols, rows, col_point, row_point, tables = cell[0][1], cell[1], cell[2], cell[3], cell[4], cell[5]
    print(pos)
    table_im = ori_img[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
    # table_line_regions = text_predict(table_im, 1, 1, table_im)

    # word_list = []
    # for region in table_line_regions:
    #     region_y = [region[0][1], region[0][5]]
    #     region_x = [region[0][0], region[0][2]]
    #     content = predict(Image.fromarray(region[1]).convert('L'))
    #     content = (content[0][0], content[0][1], content[1])
    #     for indexi, cont in enumerate(content[1]):
    #         if cont[0] > 0.9:
    #             content[0][indexi] = content[0][indexi][0]
    #             content[1][indexi] = [-1]
    #     while 1:
    #         try:
    #             content[1].remove([-1])
    #         except:
    #             break
    #     x = content[2]
    #     content = calculate(content)
    #     for index, word in enumerate(content):
    #         word_list.append(
    #             [[x[index][0] + region_x[0], region_y[0], x[index][1] + region_x[0], region_y[0], x[index][0]
    #               + region_x[0], region_y[1], x[index][1] + region_x[0], region_y[1]], word])
    # print(word_list)
    # for region in table_line_regions:
    #     cv2.rectangle(table_im, (region[0][0], region[0][1]), (region[0][6], region[0][7]), (0, 255, 0), 1)
    # for i in word_list:
    #     cv2.rectangle(table_im, (i[0][0], i[0][1]), (i[0][6], i[0][7]), (255, 0, 0), 1)
    #
    Image.fromarray(table_im).save('single_table.jpg')

    col_point = sorted(col_point)
    row_point = sorted(row_point)
    # tables = sorted(tables, key=lambda i: i[1][3], reverse=True)
    tables = sorted(tables, key=lambda i: i[1][0] + i[1][1])
    # print('sorted_tables:', tables)
    for i in tables:
        d = {'col_begin': 0, 'col_end': 0, 'row_begin': 0, 'row_end': 0}
        # print(i[1])
        for index, value in enumerate(col_point):
            if index == 0:
                d_range = 50
            else:
                d_range = (col_point[index] - col_point[index - 1]) / 3
            if i[1][0] > col_point[index] - d_range:
                # print(33333333333, i[1], index)
                d['col_begin'] = index
        for index, value in enumerate(col_point):
            if index == len(col_point) - 1:
                d_range = 50
            else:
                d_range = (col_point[index + 1] - col_point[index]) / 3
            if i[1][0] + i[1][2] < col_point[index] + d_range:
                d['col_end'] = index
                break
        for index, value in enumerate(row_point):
            if index == 0:
                d_range = 0
            else:
                d_range = (row_point[index] - row_point[index - 1]) / 3
            if i[1][1] > row_point[index] - d_range:
                d['row_begin'] = index
        for index, value in enumerate(row_point):
            if index == len(row_point) - 1:
                d_range = 0
            else:
                d_range = (row_point[index + 1] - row_point[index]) / 3
            if i[1][1] + i[1][3] < row_point[index] + d_range:
                d['row_end'] = index
                break
        # print(d)
        i.append(d)

    for i in tables:
        # cell_region = [i[1][0], i[1][1], i[1][0] + i[1][2], i[1][1] + i[1][3]]
        word_str = []
        # for word in word_list:
        #     word_center_point = ((word[0][0] + word[0][2]) / 2, (word[0][1] + word[0][5]) / 2)
        #     if cell_region[0] < word_center_point[0] < cell_region[2] and cell_region[1] < word_center_point[1] < \
        #             cell_region[3]:
        #         word_str.append(word)
        word_str = sorted(word_str, key=lambda x: x[0][1])
        word_lines = []
        word_temp = []
        for index, word in enumerate(word_str):
            if len(word_temp) == 0:
                word_temp.append(word)
                continue
            if word[0][1] == word_temp[-1][0][1]:
                word_temp.append(word)
            else:
                word_temp = sorted(word_temp, key=lambda x: x[0][0])
                word_lines.append(word_temp)
                word_temp = [word]
            if index == len(word_str) - 1:
                if len(word_temp) != 0:
                    word_lines.append(word_temp)
        word_str = ''
        for line in word_lines:
            for word in line:
                word_str += word[1]
        i.append([word_str, i[1][2], i[1][3]])

    new_table = []
    for i in tables:
        new_table.append([i[2], i[3]])

    return new_table, rows, cols, pos


if __name__ == '__main__':
    
    test_dir = r'predict'
    doc_dir = r'word'
    for file in os.listdir(test_dir):
        # if not file.endswith('docx') and not any(word in file for word in ['mask', 'predict']):
        if file == '1.jpg':
            img_path = os.path.join(test_dir, file)
            img = Image.open(img_path).convert('RGB')

            print(file)
            img = np.array(img)

            tables_info = extract_table(img, file)
            if not tables_info:
                print('Table info error!')
                continue

            doc = Document()
            # try:
            for table_info in tables_info:
                i = ['table', generate_table(tables_info[0], img)]

                i = [i, 'table', 10.5, 1, 0]
                i = [i[0][1], i[1], i[2], i[3], i[4]]

                doc = restore_table(doc, i, Image.fromarray(img))
            doc_file = os.path.splitext(file)[0] + '.docx'
            doc.save(os.path.join(doc_dir, doc_file))
            # except Exception as e:
            #     print(e)



