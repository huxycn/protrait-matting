#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -----------------------------------------------------------------------------
#
# Prepare for Portrait Matting
#
#   Create alpha weight image
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


def create_pseudo_alpha(name, src_dir, dst_dir):
    src_path = os.path.join(src_dir, name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_path = os.path.join(dst_dir, name)
    if os.path.exists(dst_path):
        logger.info('"%s" exists, Skip...', dst_path)
        return

    logger.info('Pseudo alpha from "%s"', src_path)
    mask = cv2.imread(src_path)
    alpha = cv2.GaussianBlur(mask, (51, 51), 0)

    # Save
    cv2.imwrite(dst_path, alpha)


class AlphaWeightLut(object):
    def __init__(self, names, alpha_dir, n_data_use=300):
        sum_distrib = None

        n_data = len(names)
        for idx in np.random.permutation(n_data)[:n_data_use]:
            # Load alpha image
            path = os.path.join(alpha_dir, names[idx])
            if not os.path.exists(path):
                logger.error('"%s" dose not exist', path)
                continue
            alpha = cv2.imread(path, 0)

            # Histogram
            distrib, _ = np.histogram(alpha, bins=256)

            # Sum up
            if sum_distrib is None:
                sum_distrib = distrib
            else:
                sum_distrib += distrib

        # Convert to information content
        self.distrib_lut = -np.log(sum_distrib / np.sum(sum_distrib))

    def lookup(self, alpha):
        assert alpha.dtype == np.uint8
        return self.distrib_lut[alpha]


def compute_weights(name, src_dir, dst_dir, weight_lut):
    src_path = os.path.join(src_dir, name)
    if not os.path.exists(src_path):
        logger.error('"%s" dose not exist', src_path)
        return

    dst_path = os.path.join(dst_dir, name + '.npz')
    if os.path.exists(dst_path):
        logger.info('"%s" exists, Skip...', dst_path)
        return

    logger.info('Alpha weight for "%s"', src_path)
    alpha = cv2.imread(src_path, 0)
    assert alpha.ndim == 2
    weight = weight_lut.lookup(alpha)

    # Cast for saving storage
    weight = weight.astype(np.float32)

    # Save
    np.savez(dst_path, weight=weight)


def main():
    conf = DefaultConfig()

    if conf.pseudo_alpha:
        logger.info('Compute pseudo alpha images')
        # Get valid names in 6 channel segmentation stage
        names = get_valid_names(conf.img_crop_dir, conf.img_mask_dir,
                                conf.img_mean_mask_dir,
                                conf.img_mean_grid_dir,
                                rm_exts=[False, False, False, True])
        # Create pseudo alpha images
        os.makedirs(conf.img_alpha_dir, exist_ok=True)
        for name in names:
            create_pseudo_alpha(name, conf.img_mask_dir,
                                conf.img_alpha_dir)

    # Get valid names for alpha matting
    names = get_valid_names(conf.img_crop_dir, conf.img_mask_dir,
                            conf.img_mean_mask_dir, conf.img_mean_grid_dir,
                            conf.img_alpha_dir,
                            rm_exts=[False, False, False, True, False])

    # Pre-compute look up table for weights
    logger.info('Compute look up table for weights')
    weight_lut = AlphaWeightLut(names, conf.img_alpha_dir)

    # Compute weight matrix
    logger.info('Compute weight matrix for each image')
    os.makedirs(conf.img_alpha_weight_dir, exist_ok=True)
    for name in names:
        compute_weights(name, conf.img_alpha_dir,
                        conf.img_alpha_weight_dir, weight_lut)


if __name__ == '__main__':
    main()
