#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
#
# Prepare for Trimap Segmentation
#
#   Create trimap from alpha
#
# -----------------------------------------------------------------------------

import argparse
import os
import cv2
import numpy as np
import scipy.sparse

# modules
from scripts import log_initializer
from config import DefaultConfig
from scripts.datasets import get_valid_names

# logging
from logging import getLogger, INFO
log_initializer.set_fmt()
log_initializer.set_root_level(INFO)
logger = getLogger(__name__)


def compute_trimap_from_alpha(name, src_dir, dst_dir, open_size=10,
                              alpha_margin=10):
    src_path = os.path.join(src_dir, name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_path = os.path.join(dst_dir, name)
    if os.path.exists(dst_path):
        logger.info('"%s" exists, Skip...', dst_path)
        return

    logger.info('Trimap for "%s"', src_path)
    alpha = cv2.imread(src_path, 0)
    assert alpha.ndim == 2

    # Compute each region
    fore = ((255 - alpha_margin) < alpha)
    back = (alpha < alpha_margin)
    unknown = ~(fore + back)
    unknown = cv2.dilate(
        unknown.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    ).astype(np.bool)

    # Draw
    trimap = np.zeros_like(alpha)
    trimap[fore] = 255
    trimap[unknown] = 127

    cv2.imwrite(dst_path, trimap)


def main():
    # Argument
    conf = DefaultConfig()
    # Get valid names for alpha matting
    names = get_valid_names(conf.img_crop_dir, conf.img_mask_dir,
                            conf.img_mean_mask_dir, conf.img_mean_grid_dir,
                            conf.img_alpha_dir,
                            rm_exts=[False, False, False, True, False])

    # Compute trimap
    logger.info('Compute weight matrix for each image')
    os.makedirs(conf.img_trimap_dir, exist_ok=True)
    for name in names:
        compute_trimap_from_alpha(name, conf.img_alpha_dir,
                                  conf.img_trimap_dir)


if __name__ == '__main__':
    main()
