# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.


import os

import argparse
import numpy as np
import PIL.Image as pil

from .utils import readlines

def export_gt_depths_nyuv2():

    parser = argparse.ArgumentParser(description='export_gt_depth')

    parser.add_argument('--data_path',
                        type=str,
                        help='path to the root of the NYUv2 data',
                        required=True)
    parser.add_argument('--split',
                        type=str,
                        help='which split to export gt from',
                        choices=["nyuv2"])
    opt = parser.parse_args()

    split_folder = os.path.join("splits", opt.split)
    lines = readlines(os.path.join(split_folder, "test_files.txt"))

    print("Exporting ground truth depths for {}".format(opt.split))

    gt_depths = []
    for line in lines:

        folder, frame_id = line.split()
        frame_id = frame_id

        gt_depth_path = os.path.join(opt.data_path, folder, f"{frame_id}_depth.png")
        gt_depth = np.array(pil.open(gt_depth_path)).astype(np.float32) / 255

        gt_depths.append(gt_depth.astype(np.float32))

    output_path = os.path.join(split_folder, "gt_depths.npz")

    print("Saving to {}".format(opt.split))

    np.savez_compressed(output_path, data=np.array(gt_depths))


if __name__ == "__main__":
    export_gt_depths_nyuv2()
