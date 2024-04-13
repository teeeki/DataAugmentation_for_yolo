import numpy as np
import cv2
import os

from lib.helpers import *

class yoloRotatebbox:
    def __init__(self, img_name, label_name, angle):
        

        self.img_name = img_name
        self.label_name = label_name
        self.angle = angle

        # Read image using cv2
        self.image = cv2.imread(self.img_name, 1)
        rotation_angle = self.angle * np.pi / 180
        self.rot_matrix = np.array(
            [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])

    def rotateYolobbox(self):

        new_height, new_width = self.rotate_image().shape[:2]

        f = open(self.label_name, 'r')

        f1 = f.readlines()

        new_bbox = []

        H, W = self.image.shape[:2]

        for x in f1:
            bbox = x.strip('\n').split(' ')
            if len(bbox) > 1:
                (center_x, center_y, bbox_width, bbox_height) = yoloFormattocv(float(bbox[1]), float(bbox[2]),
                                                                               float(bbox[3]), float(bbox[4]), H, W)

                upper_left_corner_shift = (center_x - W / 2, -H / 2 + center_y)
                upper_right_corner_shift = (bbox_width - W / 2, -H / 2 + center_y)
                lower_left_corner_shift = (center_x - W / 2, -H / 2 + bbox_height)
                lower_right_corner_shift = (bbox_width - W / 2, -H / 2 + bbox_height)

                new_lower_right_corner = [-1, -1]
                new_upper_left_corner = []

                for i in (upper_left_corner_shift, upper_right_corner_shift, lower_left_corner_shift,
                          lower_right_corner_shift):
                    new_coords = np.matmul(self.rot_matrix, np.array((i[0], -i[1])))
                    x_prime, y_prime = new_width / 2 + new_coords[0], new_height / 2 - new_coords[1]
                    if new_lower_right_corner[0] < x_prime:
                        new_lower_right_corner[0] = x_prime
                    if new_lower_right_corner[1] < y_prime:
                        new_lower_right_corner[1] = y_prime

                    if len(new_upper_left_corner) > 0:
                        if new_upper_left_corner[0] > x_prime:
                            new_upper_left_corner[0] = x_prime
                        if new_upper_left_corner[1] > y_prime:
                            new_upper_left_corner[1] = y_prime
                    else:
                        new_upper_left_corner.append(x_prime)
                        new_upper_left_corner.append(y_prime)
                #             print(x_prime, y_prime)

                new_bbox.append([bbox[0], new_upper_left_corner[0], new_upper_left_corner[1],
                                 new_lower_right_corner[0], new_lower_right_corner[1]])

        return new_bbox

    def rotate_image(self):
        """
        Rotates an image (angle in degrees) and expands image to avoid cropping
        """
        height, width = self.image.shape[:2]  # image shape has 3 dimensions
        image_center = (width / 2,
                        height / 2)  # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

        rotation_mat = cv2.getRotationMatrix2D(image_center, self.angle, 1.)

        # rotation calculates the cos and sin, taking absolutes of those.
        abs_cos = abs(rotation_mat[0, 0])
        abs_sin = abs(rotation_mat[0, 1])

        # find the new width and height bounds
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)

        # subtract old image center (bringing image back to origin) and adding the new image center coordinates
        rotation_mat[0, 2] += bound_w / 2 - image_center[0]
        rotation_mat[1, 2] += bound_h / 2 - image_center[1]

        # rotate image with the new bounds and translated rotation matrix
        rotated_mat = cv2.warpAffine(self.image, rotation_mat, (bound_w, bound_h))
        return rotated_mat


def func(img_name, label_name, file, OUT_DIR_IMAGES, OUT_DIR_LABELS):
    img_augs = []
    num = 0
    
    # データ拡張のバリエーションを増やしたい場合は以下のangleリストに回転角度を追加する
    angle = [45, -45, 15]
    for i in range(len(angle)): # angleリストの長さ分ループ
        im = yoloRotatebbox(img_name, label_name, angle[i])
        bbox = im.rotateYolobbox()
        # img_augs.append(im.rotate_image())
        img = im.rotate_image()
        file_path_img = OUT_DIR_IMAGES + file + "_" + "rotate" + str(num) + ".jpg"
        result = cv2.imwrite(file_path_img, img)
        if not result:
            print("画像が保存できない")
            exit(-1)
        print(f"Succesfull image : {file_path_img}")
        
        after_path_label = OUT_DIR_LABELS + file + "_" + "rotate" + str(num) + ".txt"
        formatted_string = []
        for n in bbox:
            with open(after_path_label, 'w') as fout:
                formatted_string.append(' '.join(map(str, cvFormattoYolo(n, im.rotate_image().shape[0], im.rotate_image().shape[1]))) + '\n')
                fout.writelines(formatted_string)
        num += 1  