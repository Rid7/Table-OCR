import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import cv2
import numpy as np

from PIL import Image
from docx import Document

from PSE.predict import PSE_model
from PSE.config import config as text_detection_conf
from crnn.predict import Crnn_model
from crnn.config import config as text_recognition_conf
# from ctpn.ctpn_blstm_test_full import text_predict
# from densent_ocr.model import predict
# from crnn_seq2seq_ocr.inference import attention
from restore_table import restore_table
from viterbi import calculate

text_detection_model = text_detection_conf.model_path
text_detection_device = text_detection_conf.GPU_ID
text_detection = PSE_model(model_path=text_detection_model, gpu_id=text_detection_device)

text_recognition_model = text_recognition_conf.model_path
text_recognition_device = text_recognition_conf.GPU_ID
text_recognition = Crnn_model(model_path=text_recognition_model, gpu_id=text_recognition_device)

DEBUG = True


def save_result(src, region_info, color, thickness, save_dir, file):
    for info in region_info:
        xmin, ymin = info[0][0], info[0][1]
        xmax, ymax = info[0][6], info[0][7]
        cv2.rectangle(src, (xmin, ymin), (xmax, ymax), color, thickness)
    cv2.imwrite('{0}/{1}'.format(save_dir, file), src)


def show_result(src, window_name):
    if DEBUG:
        cv2.namedWindow(window_name, 0)
        cv2.imshow(window_name, src)
        cv2.waitKey(0)


class TableConstruct:
    def __init__(self, image, file):
        """
        :param image: original image: RGB mode
        :param line_img: table line image: RGB mode but but in binary color
        :param file: dealing file name
        """
        self.image = image
        # self.line_img = line_img
        self.file = file

    def remove_line(self, src, FLAG, max_depth=10, gap_rate=0.05, epsilon=1e-6):
        """ remove noise lines
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

                    if abs(horizontal_left[1][0] - horizontal_left[0][0]) / (
                            horizontal_left[1][0] + epsilon) > gap_rate:
                        x, y, w, h = horizontal_left[0]
                        src[y:y + h, x:x + w] = 0

                    elif abs(horizontal_right[-1][0] + horizontal_right[-1][2] - \
                             (horizontal_right[-2][0] + horizontal_right[-2][2])) / \
                            (horizontal_right[-1][0] + horizontal_right[-1][2] + epsilon) > gap_rate:
                        x, y, w, h = horizontal_right[-1]
                        src[y:y + h, x:x + w] = 0
                    else:
                        return src
                else:
                    return src
                max_depth = max_depth - 1
                return self.remove_line(src, FLAG=0, max_depth=max_depth)

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
                        src[y:y + h, x:x + w] = 0

                    elif abs(vertical_down[-1][1] + vertical_down[-1][3] - \
                             (vertical_down[-2][1] + vertical_down[-2][3])) / \
                            (vertical_down[-1][1] + vertical_down[-1][3] + epsilon) > gap_rate:

                        x, y, w, h = vertical_down[-1]
                        src[y:y + h, x:x + w] = 0
                    else:
                        return src
                else:
                    return src
                max_depth = max_depth - 1
                return self.remove_line(src, FLAG=1, max_depth=max_depth)
        else:
            return src

    def get_roi(self, src, FLAG, mask):
        """get table region and coordinate from source image
        :param src: Source image: RGB mode
        :param FLAG: 0 for the 1st time getting roi, 1 for the 2nd time
        :param mask: Line image: threshold mode
        :return: rois: rois[i][0]: region of interest in the source image: RGB mode,
        rois[i][1]: roi coordinates of source image : top-left x, top-left y, width, height
        """
        mark_src = src.copy()
        if FLAG == 0:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mark_src, contours, -1, (255, 0, 0), 2)
            show_result(mark_src, 'RETR_EXTERNAL')
        elif FLAG == 1:
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(mark_src, contours, -1, (255, 0, 0), 2)
            show_result(mark_src, 'RETR_TREE')

        contours_poly = [''] * len(contours)
        rois = []
        for i, j in zip(range(len(contours)), hierarchy[0]):
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
                cv2.rectangle(mark_src, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if any([h < 15, w < 15, min_rect_h < 15, min_rect_w < 15, area < 100]):
                    continue

                roi = src[y:y + h, x:x + w]
                rois.append([roi, [x, y, w, h]])

            elif FLAG == 1 and j[3] > 0:
                contours_poly[i] = cv2.approxPolyDP(np.array(contours[i]), 3, True)
                x, y, w, h = cv2.boundingRect(np.array(contours_poly[i]))
                cv2.rectangle(mark_src, (x, y), (x + w, y + h), (0, 0, 255), 2)
        show_result(mark_src, 'Cell')
        return rois

    def rotate_degree(self, vertical):
        """calculate rotate degree according to vertical Houghline
        :param vertical: Vertical line image：threshold mode
        :return: if vertical image and Houghline detecting result exists: return the calculate degree,
        else return the default 0
        """
        mark_src = self.image.copy()
        if len(np.nonzero(vertical)) != 1:
            lines = cv2.HoughLinesP(vertical, 2, np.pi / 180, 150, minLineLength=50, maxLineGap=30)

            angles = []
            if len(np.nonzero(lines)) != 1:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.linalg.norm(np.array([[x1, y1], [x2, y2]], dtype=np.float32))
                    if length / vertical.shape[0] > 0.2:
                        angle = np.arcsin((x1 - x2) / length)
                        angles.append(angle)
                        cv2.line(mark_src, (x1, y1), (x2, y2), (0, 0, 255), 2)
            else:
                return 0

            show_result(mark_src, 'Houghline')

            mean_angle = np.mean(angles)
            degree = mean_angle * (180.0 / np.pi)
            return degree
        return 0

    def warp_perspective(self, src, vertical, horizontal):
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

        # contours, hierarchy = cv2.findContours(vertical, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(vertical, contours, -1, (255,255,255), -1)
        # contours, hierarchy = cv2.findContours(horizontal, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(horizontal, contours, -1, (255,255,255), -1)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        raw_approx = cv2.approxPolyDP(contours[0], 10, True)
        raw_hull = cv2.convexHull(raw_approx, clockwise=True)
        poly_image = np.zeros_like(mark_src) + 255
        cv2.polylines(poly_image, [raw_hull], True, (0, 0, 0), 2)
        poly_image = cv2.cvtColor(poly_image, cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(poly_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx = cv2.approxPolyDP(contours[0], 10, True)
        hull = cv2.convexHull(approx, clockwise=True)
        cv2.polylines(mark_src, [hull], True, (0, 0, 255), 2)
        show_result(mark_src, 'Poly_hull')

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

        dst_corner = np.array(((0, 0),
                               (width - 1, 0),
                               (width - 1, height - 1),
                               (0, height - 1)),
                              np.float32)

        M, _ = cv2.findHomography(sorted_corner, dst_corner, cv2.RANSAC, 10)
        src = cv2.warpPerspective(src, M, (width, height))
        vertical = cv2.warpPerspective(vertical, M, (width, height))
        horizontal = cv2.warpPerspective(horizontal, M, (width, height))
        processed = True
        print('warp perspective done')
        return src, vertical, horizontal, processed

    def table_lines(self, src, horizontal=[], vertical=[]):
        """
        :param src: 1st roi in Source image: RGB mode
        :param horizontal: 1st roi in horzontal line image: threshold mode
        :param vertical: 1st roi in vertical line image: threshold mode
        :return: cols: amount of columns, rows: amount of rows,
        col_points: x-coordinate of columns, row_points: y-coordinate of rows,
        cells: table cells region from the src.
        """
        src, vertical, horizontal, processed = self.warp_perspective(src, vertical, horizontal)
        if not processed:
            degree = self.rotate_degree(vertical)
            print('second stage degree: ', degree)
            if abs(degree) < 10:
                horizontal = np.array(Image.fromarray(horizontal).rotate(degree))
                vertical = np.array(Image.fromarray(vertical).rotate(degree))
                src = np.array(Image.fromarray(src).rotate(degree))
                self.image = np.array(Image.fromarray(self.image).rotate(degree))
                # self.line_img = np.array(Image.fromarray(self.line_img).rotate(degree))

        h_x, h_y, h_w, h_h = cv2.boundingRect(horizontal)
        v_x, v_y, v_w, v_h = cv2.boundingRect(vertical)

        OFFSET = 5
        cv2.line(vertical, (h_x + OFFSET, h_y + OFFSET), (h_x + OFFSET, h_y + h_h - OFFSET), 255, 3)
        cv2.line(vertical, (h_x + h_w - OFFSET, h_y + OFFSET), (h_x + h_w - OFFSET, h_y + h_h - OFFSET), 255, 3)
        cv2.line(horizontal, (v_x + OFFSET, v_y + OFFSET), (v_x + v_w - OFFSET, v_y), 255, 3)
        cv2.line(horizontal, (v_x + OFFSET, v_y + v_h - OFFSET), (v_x + v_w - OFFSET, v_y + v_h - OFFSET), 255, 3)

        mask = horizontal + vertical
        show_result(vertical, 'Repair_border')

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

        cells = self.get_roi(src, 1, mask)
        show_result(mask, 'Table_ROI')

        joints_contours, _ = cv2.findContours(joints, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

        cols, rows, col_points, row_points = len(xs) - 1, len(ys) - 1, list(xs), list(ys)
        return cols, rows, col_points, row_points, cells

    def draw_line(self, scale=30):
        """
        :param scale: kernel size = line image h or w / scale
        :return: rois: [table_i:[array of region of interest in the source image, [top-left_x,top-left_y, width, height]
        horizontal: horizontal lines image: thresh mode
        vertical: vertical lines image: thresh mode
        """
        # if not (self.line_img.data or self.image.data):
        if not self.image.data:
            print('Not picture!')
            return [], [], []

        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # erode_size = 3
        # element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_size, erode_size), (-1, -1))
        # gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, element)

        thresh = cv2.adaptiveThreshold(~gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -10)
        show_result(thresh, 'Thresh')
        horizontal = thresh
        vertical = thresh

        horizontalsize = int(horizontal.shape[0] / scale)
        horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontalsize, 1))
        horizontal = cv2.erode(horizontal, horizontalStructure, (-1, -1), iterations=3)
        horizontal = cv2.dilate(horizontal, horizontalStructure, (-1, -1), iterations=3)
        horizontal = cv2.blur(horizontal, (5, 5))

        verticalsize = int(vertical.shape[1] / scale)
        verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
        vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
        vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))
        vertical = cv2.blur(vertical, (5, 5))

        show_result(vertical, 'Vertical')
        show_result(horizontal, 'Horizontal')

        degree = self.rotate_degree(vertical)
        print('first stage degree:', degree)
        # horizontal = np.array(Image.fromarray(horizontal).rotate(degree))
        # vertical = np.array(Image.fromarray(vertical).rotate(degree))
        # self.image = np.array(Image.fromarray(self.image).rotate(degree))
        # self.line_img = np.array(Image.fromarray(self.line_img).rotate(degree))

        if abs(degree) < 1.5:
            horizontal = self.remove_line(horizontal, FLAG=0, gap_rate=0.05)
            vertical = self.remove_line(vertical, FLAG=1, gap_rate=0.05)

        show_result(horizontal, 'Denoised_horizontal')
        show_result(vertical, 'Denoised_vertical')

        mask = horizontal + vertical

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

        if len(xs) <= 2:
            print('Joints not satisfied！')
            return [], horizontal, vertical

        rois = self.get_roi(src=self.image, FLAG=0, mask=mask)
        return rois, horizontal, vertical

    def extract_table(self):
        """
        :return: cells: cells[:][0]: all table roi of image
        cells[:][1]: amount of table columns
        cells[:][2]: amount of table rows
        cells[:][3]: x-coordinates of columns
        cells[:][4]: y-coordinate of rows,
        cells[:][5]: table cells region from the src.
        """
        image = self.image.copy()
        print(image.shape)

        table_rois, horizontal, vertical = self.draw_line()

        if not table_rois:
            return None

        table_rois = sorted(table_rois, key=lambda i: i[1][0] + i[1][1])  # sort tables by sum of top-left x,y
        cells = []
        for table_roi in table_rois:
            table_image = table_roi[0]
            x, y, w, h = table_roi[1]
            cols, rows, col_point, row_point, blocks = self.table_lines(src=table_image,
                                                                        horizontal=horizontal[y:y + h, x:x + w],
                                                                        vertical=vertical[y:y + h, x:x + w])
            print('cols and rows', cols, rows)

            if cols > 1 and rows > 1:
                cells.append([table_roi, cols, rows, col_point, row_point, blocks])
                show_result(table_image, 'Final table')
                cv2.imwrite('table/{}'.format(self.file), table_image)


        if cells:
            return cells
        else:
            return None

    def generate_table(self, cell):
        def closest_index(points: list, target):
            for index, value in enumerate(points):
                if abs(value - target) < abs(points[index - 1] - target) and index > 0 or index == 0: closest = index
            return closest

        pos, cols, rows, col_point, row_point, tables = cell[0][1], cell[1], cell[2], cell[3], cell[4], cell[5]

        table_im = self.image[pos[1]:pos[1] + pos[3], pos[0]:pos[0] + pos[2]]
        table_copy = table_im.copy()

        text_rects = text_detection.predict(table_im, 1, 1, table_copy)

        char_info = []
        for text_rect in text_rects:
            region_y = [text_rect[0][1], text_rect[0][5]]  # text_rectangle_ymin,  text_rectangle_ymax
            region_x = [text_rect[0][0], text_rect[0][2]]  # text_rectangle_xmin,  text_rectangle_xmax

            char_list, prob_list, char_positions = text_recognition.predict(Image.fromarray(text_rect[1]).convert('L'))

            for index, top5_confidence in enumerate(prob_list):
                if top5_confidence[0] > 0.5:
                    char_list[index] = char_list[index][0]  # get the top1 char recognition result if confidence > 50%
                    prob_list[index] = [-1]  # then set the confirmed char confidence to -1

            prob_list = list(filter(lambda x: x[0] != -1, prob_list))

            content = [char_list, prob_list, char_positions]
            content = calculate(content)  # replace low confidence char recognition result by edit distance
            for index, char in enumerate(content):
                char_left, char_right = char_positions[index]
                char_info.append([[char_left + region_x[0], region_y[0],
                                   char_right + region_x[0], region_y[0],
                                   char_left + region_x[0], region_y[1],
                                   char_right + region_x[0], region_y[1]], char])

        for text_rect in text_rects:
            xmin, ymin = text_rect[0][0], text_rect[0][1]
            xmax, ymax = text_rect[0][6], text_rect[0][7]
            cv2.rectangle(table_copy, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        save_result(table_copy, char_info, color=(255, 0, 0), thickness=2, save_dir='char_position', file=self.file)

        col_point = sorted(col_point)
        row_point = sorted(row_point)

        for table in tables:
            cell_dict = {'col_begin': 0, 'col_end': 0, 'row_begin': 0, 'row_end': 0}
            cell_x, cell_y, cell_w, cell_h = table[1]
            cell_dict['col_begin'] = closest_index(col_point, cell_x)
            cell_dict['col_end'] = closest_index(col_point, cell_x + cell_w)
            cell_dict['row_begin'] = closest_index(row_point, cell_y)
            cell_dict['row_end'] = closest_index(row_point, cell_y + cell_h)
            table.append(cell_dict)

            cell_char = []
            for info in char_info:
                char_xmin, char_xmax, char_ymin, char_ymax = info[0][0], info[0][2], info[0][3], info[0][5]
                char_xcenter, char_ycenter = np.mean([[char_xmin, char_ymin], [char_xmax, char_ymax]], axis=0)

                if cell_x < char_xcenter < cell_x + cell_w and cell_y < char_ycenter < cell_y + cell_h:
                    cell_char.append(info)

            cell_char = sorted(cell_char, key=lambda x: x[0][1])
            print('cell_char:', cell_char)
            cell_text = []
            text_temp = []
            if len(cell_char) == 1:
                cell_text = [cell_char]
            else:
                for index, char in enumerate(cell_char):
                    if len(text_temp) == 0:
                        text_temp.append(char)
                        continue
                    if char[0][1] == text_temp[-1][0][1]:
                        text_temp.append(char)
                    else:
                        text_temp = sorted(text_temp, key=lambda x: x[0][0])
                        cell_text.append(text_temp)
                        text_temp = [char]
                    if index == len(cell_char) - 1:
                        if len(text_temp) != 0:
                            cell_text.append(text_temp)

            cell_text = "".join([char[1] for line in cell_text for char in line])
            print('cell_text：', cell_text)
            table.append([cell_text, table[1][2], table[1][3]])

        new_table = []
        for table in tables:
            new_table.append([table[2], table[3]])
        return new_table, rows, cols, pos


if __name__ == '__main__':

    test_dir = 'test'
    # line_dir = 'test'
    doc_dir = 'word'
    global file
    for file in os.listdir(test_dir):
        # if not file.endswith('pdf') and not any(word in file for word in ['mask', 'predict']):
        if file == 'test_2.jpg':
            print(file)

            img_path = os.path.join(test_dir, file)
            # line_path = os.path.join(line_dir, file)
            img = np.array(Image.open(img_path).convert('RGB'))
            # line_img = np.array(Image.open(line_path).resize((img.shape[1], img.shape[0])).convert('RGB'))

            table_construct = TableConstruct(image=img, file=file)
            tables_info = table_construct.extract_table()
            if not tables_info:
                print('Table info error!')
                continue

            doc = Document()
            for table_info in tables_info:
                final_info = [table_construct.generate_table(table_info), 'table', 10.5, 1, 0]
                doc = restore_table(doc, final_info, Image.fromarray(img))
            doc_file = os.path.splitext(file)[0] + '.docx'
            doc.save(os.path.join(doc_dir, doc_file))

            cv2.destroyAllWindows
