import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
from shutil import copyfile

# from lib import bokashi
from lib import LUT
from lib import noise
from lib import maskImage
from lib import rotate

global num
num = 1

# 二値化処理
def binarize(img_name):
    img_augs = []
    img = cv2.imread(img_name)
    if img is None:
        print("画像の読み込みに失敗")
        exit(-1)
    # 拡張を増やしたい場合は以下のコードをパラメータ変更して追加
    img_augs.append(cv2.threshold(img, 180, 255, cv2.THRESH_TOZERO)[1]) # 拡張1
    img_augs.append(cv2.threshold(img, 180, 255, cv2.THRESH_TOZERO)[1]) # 拡張2
    return img_augs
    
# ダウンサンプリング　拡大と縮小を繰り返して解像度を下げる
def down_sampling(img_name):
    img_augs = []
    img = cv2.imread(img_name)
    img_size = img.shape
    # 拡張を増やしたい場合は以下のコードをパラメータ変更して追加
    img_augs.append(cv2.resize(cv2.resize(img, (img_size[1]//5, img_size[0]//5)), (img_size[1], img_size[0])))
    return img_augs

# モルフォロジー収縮
def erode(img_name):
    img_augs = []
    # img_name = INPUT_DIR_PATH + file + ".jpg"
    img = cv2.imread(img_name)
    # さらに拡張したい場合は以下2行をさらに追加しフィルタを変更する
    filter_one = np.ones((3,3))# モルフォロジー処理のフィルタ
    img_augs.append(cv2.erode(img, filter_one))
    return img_augs

# 反転
def flip(img_name):
    img_augs = []
    img = cv2.imread(img_name)
    # これは拡張不可（変更するとラベルの座標も変更を加えないといけないため）
    img_augs.append(cv2.flip(img, 1))
    return img_augs

# 平滑化
def blur(img_name):
    img_augs = []
    img = cv2.imread(img_name)
    # さらに拡張したい場合は以下2行を追加しaverage_squareを変更
    average_square = (10,10)      
    img_augs.append(cv2.blur(img, average_square))
    return img_augs
  

def fake_method(file):
    img_name = INPUT_DIR_PATH + file + ".jpg"
    label_name = INPUT_DIR_PATH + file + ".txt"
    after_path_label = OUT_DIR_LABELS + file + "_" + str(num) + ".txt"

    img_normal = []
    img_normal.append(cv2.imread(img_name))
    save(img_normal, file, "normal")

    # 二値化
    img_aug = binarize(img_name)
    save(img_aug, file, "binarize")
    
    # ダウンサンプリング
    img_aug = down_sampling(img_name)
    save(img_aug, file, 'down_sampling')
    
    # モルフォロジー収縮
    img_aug = erode(img_name)
    save(img_aug, file, "erode")
    
    # 反転
    img_aug = flip(img_name)
    flip_save(img_aug, file, "flip")
    
    # 平滑化
    img_aug = blur(img_name)
    save(img_aug, file, "blur")
    
    # lib/LUT.pyによるガンマ変換とヒストグラム均一化
    img_aug = LUT.func(img_name)
    save(img_aug, file, "LUT")
    
    # lib.noise.pyによるノイズの付与
    img_aug = noise.func(img_name)
    save(img_aug, file, "noise")
    
    # lib.maskImage.pyによるラベル部分のランダムマスク処理
    img_aug = maskImage.read_label(img_name, label_name)
    save(img_aug, file, "mask")
    
    # lib.rotate.pyによる画像の回転
    rotate.func(img_name, label_name, file, OUT_DIR_IMAGES, OUT_DIR_LABELS)


  
def save(img_augs, file, string):
    global num
    # 画像の保存
    for img_aug in img_augs:
        file_path_img = OUT_DIR_IMAGES + file + "_" +str(num) + string + ".jpg"
        result = cv2.imwrite(file_path_img, img_aug)
        if not result:
            print("画像が保存できない")
            exit(-1)
        print(f"Succesfull image : {file_path_img}")
        
        # ラベルの保存
        # フォルダからラベルを読み込む
        label_name = INPUT_DIR_PATH + file + ".txt"
        with open(label_name, 'r') as label_file:
            # ファイルの内容を1つの文字列として抽出
            labels = label_file.read()
        
        after_path_label =  OUT_DIR_LABELS + file + "_" + str(num) + string + ".txt"
        with open(after_path_label, 'w') as new_label_file:
            new_label_file.write(labels)
            print(f"Succesfull label : {after_path_label}")
        
        num+=1


# flip用
def flip_save(img_augs, file, string):
    global num
    # 画像の保存
    for img_aug in img_augs:
        file_path_img = OUT_DIR_IMAGES + file + "_" + string + str(num) + ".jpg"
        result = cv2.imwrite(file_path_img, img_aug)
        if not result:
            print("画像が保存できない")
            exit(-1)
        print(f"Succesfull image : {file_path_img}")
        
        # ラベルの保存
        # フォルダからラベルを読み込む
        label_name = INPUT_DIR_PATH + file + ".txt"
        with open(label_name, 'r') as label_file:
            # 各行をリストとして抽出
            labels= label_file.readlines()
            
        # flip用ラベルの変換
        flipped_labels = []
        for label in labels:
            parts = label.split()
            object_class, x_center, y_center, width, height = map(float, parts)
            object_class = int(object_class)
            flipped_x_center = round(1 - x_center, 6)  # x_centerを反転
            flipped_labels.append(f"{object_class} {flipped_x_center} {y_center} {width} {height}\n")        

        after_path_label =  OUT_DIR_LABELS + file + "_" + string + str(num) + ".txt"
        with open(after_path_label, 'w') as new_flip_label_file:
            new_flip_label_file.writelines(flipped_labels)
            print(f"Succesfull label : {after_path_label}")
        
        num+=1



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ("please set data directory path.")
        print ("python inflate_images.py [input_dir] [out_dir_bokashi]")
        exit(-1)
    

    # 増幅するフォルダ名の決定
    global OUT_DIR_PATH
    OUT_DIR_PATH = "output"
    if len(sys.argv) > 2:
        # 出力ディレクトリ
        OUT_DIR_PATH = sys.argv[2]
    OUT_DIR_IMAGES = OUT_DIR_PATH + "/images/"
    OUT_DIR_LABELS = OUT_DIR_PATH + "/labels/"
    os.makedirs(OUT_DIR_IMAGES, exist_ok=True)
    os.makedirs(OUT_DIR_LABELS, exist_ok=True)

    # 処理対象ディレクトリ　アノテーション後の画像とラベルのフォルダを想定
    global INPUT_DIR_PATH
    INPUT_DIR_PATH = sys.argv[1]
    INPUT_DIR_PATH += "/"

    # フォルダ内の全ファイルを取得
    files = os.listdir(INPUT_DIR_PATH)
    # 除外ファイル
    remove_file = "classes.txt"
    files = [file for file in files if remove_file not in file]
    file_names = []
    for file in files:
        # 拡張子を除いたファイル名を取得
        file_name, _ = os.path.splitext(file)
        
        # ファイル名がリストになければ追加
        if file_name not in file_names:
            file_names.append(file_name)
    
    for file in file_names:
        fake_method(file)
    
    with open(INPUT_DIR_PATH + remove_file, "r") as rv_file:
        classes = rv_file.read()
    with open(OUT_DIR_PATH + "/" + remove_file, "w") as rv_file:
        rv_file.write(classes)