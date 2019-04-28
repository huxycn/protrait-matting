#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import scipy
import json

from config import crop_file_path
from config import raw_imgs_dir
from config import crop_imgs_dir


from config import images_mask_dir
from config import img_masks_dir

from config import img_size


def crop_img():
    print('Read crop file...')
    crop_dict = {}
    with open(crop_file_path, 'r') as f:
        for line in f.readlines():
            img_name = line.strip().split()[0]
            crop_coord = line.strip().split()[1:]
            crop_dict[img_name] = {}
            crop_dict[img_name]['y0'] = int(crop_coord[0])
            crop_dict[img_name]['y1'] = int(crop_coord[1])
            crop_dict[img_name]['x0'] = int(crop_coord[2])
            crop_dict[img_name]['x1'] = int(crop_coord[3])
    print(json.dumps(crop_dict))

    for img_name in os.listdir(raw_imgs_dir):
        raw_img_file_path = os.path.join(raw_imgs_dir, img_name)
        crop_img_file_path = os.path.join(crop_imgs_dir, img_name)

        img = cv2.imread(raw_img_file_path)
        y0 = crop_dict[img_name]['y0']
        y1 = crop_dict[img_name]['y1']
        x0 = crop_dict[img_name]['x0']
        x1 = crop_dict[img_name]['x1']
        print(y0, y1, x0, x1)
        img = img[y0:y1, x0:x1, :]
        img = cv2.resize(img, img_size)
        cv2.imwrite(crop_img_file_path, img)
        print('Crop {} to {}'.format(raw_img_file_path, crop_img_file_path))


def img_mask_mat2jpg():
    for mask_name in os.listdir(images_mask_dir):
        img_id = mask_name.split('_')[0]
        mat_file_path = os.path.join(images_mask_dir, mask_name)
        jpg_file_path = os.path.join(img_masks_dir, '{}.jpg'.format(img_id))
        img = scipy.io.loadmat(mat_file_path)['mask']
        img *= 255  # [0:1] -> [0:255]
        cv2.imwrite(jpg_file_path, img)
        print('Mask mat {} ==> Image mask {}'.format(mat_file_path, jpg_file_path))


if __name__ == '__main__':
    img_mask_mat2jpg()
