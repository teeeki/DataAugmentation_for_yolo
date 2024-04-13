#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys
import os
from shutil import copyfile
# ヒストグラム均一化(CLAHE)


def func(data_dir, file_name, out_dir_bokashi):


    # 平滑化用
    average_square = (10,10)

    # 画像の読み込み
    img_src = cv2.imread(
        "{0}/images/{1}".format(data_dir, file_name), 1)
    img_src = cv2.imread(
    "{0}/images/{1}".format(data_dir, file_name), 1)
    trans_img = []


    # 平滑化      
    trans_img.append(cv2.blur(img_src, average_square))     


    if not os.path.exists("{0}/{1}".format(data_dir, out_dir_bokashi)):
        os.makedirs("{0}/{1}".format(data_dir, out_dir_bokashi))
    if not os.path.exists("{0}/inflated_labels".format(data_dir)):
        os.makedirs("{0}/inflated_labels".format(data_dir))

    # 保存
    base = os.path.splitext(os.path.basename(file_name))[0] + "_"
    img_src.astype(np.float64)
    for i, img in enumerate(trans_img):
        new_file_name = base + 'bokashi' + str(i)
        #text_file_name = file_name.replace(".png", ".txt")
        text_file_name = file_name.replace(".jpg", ".txt")
        # pngかjpgか指定
        cv2.imwrite(
            "{0}/{1}/{2}".format(data_dir, out_dir_bokashi, new_file_name + ".jpg"), img)
        copyfile("{0}/labels/{1}".format(data_dir, text_file_name),
                 "{0}/inflated_labels/{1}".format(data_dir, new_file_name + ".txt"))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("please set data directory path.")
        print ("python inflate_images.py [input_dir] [out_dir_bokashi]")
        exit(-1)
    
    
    # 増幅するフォルダ名の決定
    out_dir_bokashi = "obj"
    if len(sys.argv) > 2:
    	out_dir_bokashi = sys.argv[2]
    
    data_dir = sys.argv[1]
    
    print('data_dir={0}'.format(data_dir))
    for _, dirs, _ in os.walk("{0}/images/".format(data_dir)):
        # for class_num,class_name in zip(dirs, classes):
        for _, _, files in os.walk("{0}/images/".format(data_dir,)):
            for file_name in files:
                print('file_name={0}\n'.format(file_name))
                main(data_dir, file_name, out_dir_bokashi)
