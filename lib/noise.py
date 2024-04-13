#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# ガウシアンノイズ
def addGaussianNoise(src):
  row,col,ch= src.shape
  mean = 0
  var = 0.1
  sigma = 15
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  noisy = src + gauss
  return noisy


def func(img_name):
    img = cv2.imread(img_name)
    img_augs = []
    # ノイズ付加
    img_augs.append(addGaussianNoise(img))
    return img_augs    