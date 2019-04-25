# -*- coding: utf-8 -*-
import json

# logging
from logging import getLogger, NullHandler
logger = getLogger(__name__)
logger.addHandler(NullHandler())

import os
root_dir = os.path.abspath(os.path.dirname(__file__))


class DefaultConfig():
    img_raw_dir = os.path.join(root_dir, "data/img_raw")
    img_crop_dir = os.path.join(root_dir, "data/img_crop")
    img_mask_dir = os.path.join(root_dir, "data/img_mask")
    img_mean_mask_dir = os.path.join(root_dir, "data/img_mean_mask")
    img_mean_grid_dir = os.path.join(root_dir, "data/img_mean_grid")
    img_alpha_dir = os.path.join(root_dir, "data/img_alpha")
    img_alpha_weight_dir = os.path.join(root_dir, "data/img_alpha_weight")
    img_trimap_dir = os.path.join(root_dir, "data/img_trimap")
    org_imgurl_filepath = os.path.join(root_dir, "data_original/alldata_urls.txt")
    org_crop_filepath = os.path.join(root_dir, "data_original/crop.txt")
    org_mask_dir = os.path.join(root_dir, "data_original/images_mask")
    face_predictor_filepath = os.path.join(root_dir, "shape_predictor_68_face_landmarks.dat")
    mean_mask_filepath = os.path.join(root_dir, "data/mean_mask.npz")

    pseudo_alpha = True
