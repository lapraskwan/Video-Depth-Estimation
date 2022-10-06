# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import os
import random
os.environ["MKL_NUM_THREADS"] = "1"  # noqa F402
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # noqa F402
os.environ["OMP_NUM_THREADS"] = "1"  # noqa F402

import numpy as np
from PIL import Image  # using pillow-simd for increased speed
import cv2

import torch
import torch.utils.data as data
from torchvision import transforms

cv2.setNumThreads(0)


def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class NYUv2_Rectified_Dataset(data.Dataset):
    """Superclass for monocular dataloaders
    """

    def __init__(self,
                 data_path,
                 filenames,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.jpg',
                 ):
        super().__init__()

        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.num_scales = num_scales

        self.interp = transforms.InterpolationMode.LANCZOS
        self.frame_idxs = frame_idxs
        assert self.frame_idxs == [0, -1], "frame_idxs must be [0, -1] for rectified nyuv2 dataset."

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                # check it isn't a blank frame - keep _aug as zeros so we can check for it
                if inputs[(n, im, i)].sum() == 0:
                    inputs[(n + "_aug", im, i)] = inputs[(n, im, i)]
                else:
                    inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        return len(self.filenames)

    def load_intrinsics(self, folder, index):
        if self.is_train:
            intrinsic_file = os.path.join(self.data_path, folder, f"{index:06}_cam.txt")
        else:
            intrinsic_file = os.path.join(self.data_path, folder, "cam.txt")

        with open(intrinsic_file, "r") as f:
            lines = f.readlines()

        fx = lines[0].split()[0]
        fy = lines[1].split()[1]
        cx = lines[0].split()[2]
        cy = lines[1].split()[2]

        K = np.array([[fx, 0., cx, 0.],
                           [0., fy, cy, 0.],
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]], dtype=np.float32)
        
        K[0, :] /= 320
        K[1, :] /= 256

        return K
    
    def index_to_folder_and_frame_idx(self, index):
        """Convert index in the dataset to a folder name, frame_idx and any other bits
        """
        line = self.filenames[index].split()
        folder = line[0]
        frame_index = int(line[1])

        return folder, frame_index
    
    def get_color(self, folder, frame_index, id, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, id))

        if do_flip:
            color = color.transpose(Image.FLIP_LEFT_RIGHT)

        return color

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "depth_gt"                              for ground truth depth maps

        <frame_id> is:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        folder, frame_index = self.index_to_folder_and_frame_idx(index)

        poses = {}
        for i in self.frame_idxs:
            try:
                inputs[("color", i, -1)] = self.get_color(folder, frame_index, i, do_flip)
            except FileNotFoundError as e:
                if i != 0:
                    # fill with dummy values
                    inputs[("color", i, -1)] = Image.fromarray(np.zeros((100, 100, 3)).astype(np.uint8))
                    poses[i] = None
                else:
                    raise FileNotFoundError(f'Cannot find frame - make sure your '
                                            f'--data_path is set correctly. {e}')

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.load_intrinsics(folder, frame_index)

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth and False:
            depth_gt = self.get_depth(folder, frame_index, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def check_depth(self):
        if self.is_train:
            return False
        return True

    def get_depth(self, folder, frame_index, do_flip):
        """Returns a numpy array of the ground truth depth map"""
        if self.is_train:
            raise NotImplementedError

        depth_gt = self.loader(os.path.join(self.data_path, folder, "depth", f"{frame_index:06}.png"))

        if do_flip:
            depth_gt = depth_gt.transpose(Image.FLIP_LEFT_RIGHT)

        depth_gt = np.array(depth_gt) / 255 * 10
        return depth_gt

    def get_image_path(self, folder, frame_index, id):
        if self.is_train:
            # Use 000000_0.jpg as the previous frame (frame -1) and 000000_1.jpg as the target frame (frame 0)
            image_path = os.path.join(self.data_path, folder, f"{frame_index:06}_{id + 1}.jpg")
        else:
            image_path = os.path.join(self.data_path, folder, f"{frame_index:06}.jpg")
        return image_path
