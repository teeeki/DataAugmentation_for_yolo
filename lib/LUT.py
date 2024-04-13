#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# ヒストグラム均一化(CLAHE)
def equalizeHistRGB(src):
    img_yuv = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])

    img_hist = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    return img_hist


def func(img_name):
    img = cv2.imread(img_name)
    img_augs = []

    # ルックアップテーブルの生成(ガンマ変更)
    min_table = 50
    max_table = 205
    diff_table = max_table - min_table
    gamma1 = 0.75
    gamma2 = 1.5
    # 拡張する場合は以下拡張1から拡張4の要領でコードを追加
    # gamma2 = 3 (拡張1)

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )
    LUT_G1 = np.arange(256, dtype = 'uint8' )
    LUT_G2 = np.arange(256, dtype = 'uint8' )
    #LUT_G3 = np.arange(256, dtype = 'uint8' ) # 拡張2


    LUTs = []

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
               
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
                                  
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # その他LUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255
        LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1) 
        LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)
        #LUT_G3[i] = 255 * pow(float(i) / 255, 1.0 / gamma2) # 拡張3



    LUTs.append(LUT_HC)
    LUTs.append(LUT_LC)
    LUTs.append(LUT_G1)
    LUTs.append(LUT_G2)
    # LUTs.append(LUT_G3) # 拡張4
    
    trans_img = []
    
    # LUT変換
    for i, LUT in enumerate(LUTs):
        img_augs.append(cv2.LUT(img, LUT))
      
    # ヒストグラム均一化
    img_augs.append(equalizeHistRGB(img))
    
    return img_augs

