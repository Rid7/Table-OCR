#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 20:43:35 2019
@author: zhu
"""
import os
import sys
import xml.etree.ElementTree as ET
import glob
import numpy as np

indir = '/home/rid/Data/text_detection/Custom/Annotations/'  # xml目录
outdir = '/home/rid/Data/text_detection/Custom/gt/'  # txt目录


def xml_to_txt(indir, outdir):
    os.chdir(indir)
    annotations = os.listdir('.')

    for i, file in enumerate(annotations):
        with open(os.path.join(outdir, os.path.splitext(file)[0] + '.txt'), 'w') as f:
            in_file = open(file)
            print(file)
            tree = ET.parse(in_file)
            root = tree.getroot()

            xn = []
            xx = []
            yn = []
            yx = []

            k = 0

            for obj in root.iter('object'):
                current = list()

                name = obj.find('name').text


                xmlbox = obj.find('bndbox')
                #                xn = xmlbox.find('xmin').text
                #                xx = xmlbox.find('xmax').text
                #                yn = xmlbox.find('ymin').text
                #                yx = xmlbox.find('ymax').text

                xn.append(xmlbox.find('xmin').text)
                xx.append(xmlbox.find('xmax').text)
                yn.append(xmlbox.find('ymin').text)
                yx.append(xmlbox.find('ymax').text)

            #                print xn
            #                f_w.write(name.encode("utf-8")+' ')
            #        f_w.write('img_release/' +file.encode("utf-8") + name.encode("utf-8") + ' ' + xn+' '+yn+' '+xx+' '+yx+' '+'\n')
            #        f_w.close()

            for obj in root.iter('object'):
                f.write(xn[k] + ',' + yn[k] + ',' + xx[k] + ',' + yn[k] + ',' +
                        xx[k] + ',' + yx[k] + ',' + xn[k] + ',' + yx[k] + ',' + name + '\n')
                k = k + 1


xml_to_txt(indir, outdir)