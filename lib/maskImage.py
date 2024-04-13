"""画像内のラベルづけされた矩形に対してランダムにマスクを描画する。

入力画像はJPEG形式であるものとする。
入力画像とラベルテキストファイルは拡張子を除き同一のファイル名であるものとする。
"""

import os
import cv2
from typing import List
import glob
import numpy as np
from lib import myClass
import fnmatch

"""パラメータ設定"""
# ランダムマスキングのパラメータ(元論文と近い値を設定)
PARAMETER = myClass.Parameter(0.5, 0.02, 0.4, 0.3, 3.3)


def yolo_format_to_cv(
        norm_bb_center_x: float,
        norm_bb_center_y: float,
        norm_bb_width: float,
        norm_bb_height: float,
        iamge_width: float,
        image_height: float) -> myClass.MyRect:
    """ラベル矩形をyolo形式からopencv形式に変換する

    Args:
        norm_bb_center_x: 画像幅に対して正規化された，ラベル矩形のx座標
        norm_bb_center_y: 画像高さに対して正規化された，ラベル矩形のy座標
        norm_bb_width: 画像幅に対して正規化された，ラベル矩形の幅
        norm_bb_height: 画像高さに対して正規化された，ラベル矩形の高さ
        iamge_width: 画像の幅
        image_height: 画像の高さ

    Return:
        opencv形式のラベル矩形
    """
    bounding_box_width = norm_bb_width * iamge_width
    bounding_box_height = norm_bb_height * image_height
    center_x = norm_bb_center_x * iamge_width
    center_y = norm_bb_center_y * image_height
    bounding_box_x = center_x - bounding_box_width / 2
    bounding_box_y = center_y - bounding_box_height / 2
    return myClass.MyRect(int(bounding_box_x), int(bounding_box_y),
                          int(bounding_box_width), int(bounding_box_height))


def write_mask_image(img, masks: List[myClass.MyRect], parameter: myClass.Parameter):
    img_augs = []
    """画像にマスクを描画して保存する
    Args:
        image: 入力画像
        masks: マスク
        filepath: マスク描画後の画像の出力パス
    """
    # マスクの描画色
    color = (0, 0, 0)
    # 入力画像をコピー
    masked_image = img.copy()
    
    # 拡張を増やしたい場合は以下1行のrange()の数字を増やす
    for i in range(3):
        for mask in masks:
            random_mask = []
            # ランダムマスキング領域算出
            eraser = myClass.RandomErase(parameter)
            erase_rect = eraser.erase(mask)
            #マスク矩形の画素値をランダムな値で埋める場合
            random_mask = np.random.randint(0, 255, (erase_rect.height, erase_rect.width, 3))
            masked_image = img.copy()
            masked_image[erase_rect.y:erase_rect.y + erase_rect.height,  erase_rect.x:erase_rect.x + erase_rect.width, :] = random_mask
        img_augs.append(masked_image)
    
    return img_augs


def read_label(img_name, label_name):
    img = cv2.imread(img_name)
    if img is None:
        print("maskImage.py : 画像の読み込みに失敗")
        exit(-1)
        
    with open(label_name, 'r') as label_file:        # 対応する画像ファイルを読み込む
        
        # 画像の縦横サイズを取得
        height, width = img.shape[:2]
        # マスク矩形
        masks: List[myClass.MyRect] = []
        # ラベルの座標を1行ずつ読み込み
        for label in label_file.readlines():
            # 改行文字を削除し，空白区切りで分割
            bounding_box = label.strip('\n').split(' ')
            # ラベル情報がある場合
            if len(bounding_box) > 1:
                # YOLO形式からopencv形式に変換(対応する画像サイズに合わせた整数値の各座標)
                rect = yolo_format_to_cv(
                    float(bounding_box[1]),
                    float(bounding_box[2]),
                    float(bounding_box[3]),
                    float(bounding_box[4]),
                    width,
                    height
                )
                masks.append(rect)          
    img_augs = write_mask_image(img, masks, PARAMETER)
    return img_augs