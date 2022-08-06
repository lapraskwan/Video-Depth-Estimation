"""
1. Load models (Pose)
2. Load dataset
3. Pass to pose model to get rotation and K
4. Plot graphs to show rotation (maybe histogram?)
5. Compute average rotation?
6. Plot graph to show relations between K and rotation like the one in paper.
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from manydepth import networks, datasets
from .utils import readlines
from .test_simple import load_and_preprocess_intrinsics
from .layers import transformation_from_parameters, rot_from_axisangle

def parse_args():
    parser = argparse.ArgumentParser(
        description='Captures video with camera and outputs the depth map.')
    
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to save the depth maps as video',
                        required=True)
    parser.add_argument("--data_path",
                        type=str,
                        help="path to the training data",
                        required=True)
    parser.add_argument("--dataset",
                        type=str,
                        help="dataset to train on",
                        default="kitti",
                        choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test",
                                "cityscapes_preprocessed", "nyuv2", "nyuv2_50k"])
    parser.add_argument("--split",
                        type=str,
                        help="which training split to use",
                        default="eigen_zhou")
    parser.add_argument("--batch_size",
                        type=int,
                        help="batch size",
                        default=12)
    parser.add_argument("--png",
                        help="if set, trains from raw KITTI png files (instead of jpgs)",
                        action="store_true")
    parser.add_argument("--height",
                        type=int,
                        help="input image height",
                        default=192)
    parser.add_argument("--width",
                        type=int,
                        help="input image width",
                        default=640)
    parser.add_argument("--frame_ids",
                        nargs="+",
                        type=int,
                        help="frames to load",
                        default=[0, -1, 1])
    parser.add_argument("--num_workers",
                        type=int,
                        help="number of dataloader workers",
                        default=12)
    return parser.parse_args()

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def load_models(model_path):
    """
    Load pretrained models. 
    Return: encoder, depth_decoder, pose_enc, pose_dec, encoder_dict (for width, height, min/max depth bin)
    """
    assert model_path is not None, \
        "You must specify the --model_path parameter"

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("-> Loading model from ", model_path)

    # Loading pretrained model
    print("   Loading pretrained encoder")
    encoder_dict = torch.load(os.path.join(model_path, "encoder.pth"), map_location=device)
    encoder = networks.ResnetEncoderMatching(18, False,
                                             input_width=encoder_dict['width'],
                                             input_height=encoder_dict['height'],
                                             adaptive_bins=True,
                                             min_depth_bin=encoder_dict['min_depth_bin'],
                                             max_depth_bin=encoder_dict['max_depth_bin'],
                                             depth_binning='linear',
                                             num_depth_bins=96)

    filtered_dict_enc = {k: v for k, v in encoder_dict.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(os.path.join(model_path, "depth.pth"), map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    print("   Loading pose network")
    pose_enc_dict = torch.load(os.path.join(model_path, "pose_encoder.pth"),
                               map_location=device)
    pose_dec_dict = torch.load(os.path.join(model_path, "pose.pth"), map_location=device)

    pose_enc = networks.ResnetEncoder(18, False, num_input_images=2)
    pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1,
                                    num_frames_to_predict_for=2)

    pose_enc.load_state_dict(pose_enc_dict, strict=True)
    pose_dec.load_state_dict(pose_dec_dict, strict=True)

    # Setting states of networks
    encoder.eval()
    depth_decoder.eval()
    pose_enc.eval()
    pose_dec.eval()
    if torch.cuda.is_available():
        encoder.cuda()
        depth_decoder.cuda()
        pose_enc.cuda()
        pose_dec.cuda()

    # return encoder, depth_decoder, pose_enc, pose_dec, encoder_dict
    return pose_enc, pose_dec

def load_dataset(args):
    frames_to_load = args.frame_ids.copy()
    datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                        "cityscapes_preprocessed": datasets.CityscapesPreprocessedDataset,
                        "kitti_odom": datasets.KITTIOdomDataset,
                        "nyuv2": datasets.NYUv2RawDataset,
                        "nyuv2_50k": datasets.NYUv2_50K_Dataset}
    dataset = datasets_dict[args.dataset]

    assert os.path.isdir(os.path.join("splits", args.split)), \
        f"{args.split} split does not exist."

    fpath = os.path.join("splits", args.split, "{}_files.txt")
    train_filenames = readlines(fpath.format("train"))
    img_ext = '.png' if args.png else '.jpg'

    train_dataset = dataset(
        args.data_path, train_filenames, args.height, args.width,
        frames_to_load, 4, is_train=True, img_ext=img_ext)
    train_loader = DataLoader(
        train_dataset, args.batch_size, True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        worker_init_fn=seed_worker)

    num_train_samples = len(train_filenames)
    print(f"There are {num_train_samples} training samples.")

    return train_loader

def predict_pose(inputs, pose_enc, pose_dec, args):
    """
    Return a list of predicted poses.
    len(output) = len(frame_ids) - 1
    outputs: [{"id", _, "axisangle": _, "translation": _, "K": _, "cam_T_cam": _}, {...}, ]
    """
    outputs = []
    # In this setting, we compute the pose to each source frame via a
    # separate forward pass through the pose network.

    # predict poses for reprojection loss
    # select what features the pose network takes as input
    pose_feats = {f_i: inputs["color_aug", f_i, 0] for f_i in args.frame_ids}
    for f_i in args.frame_ids[1:]:
        if f_i != "s":
            frame_output = {"id": f_i}

            # To maintain ordering we always pass frames in temporal order
            if f_i < 0:
                pose_inputs = [pose_feats[f_i], pose_feats[0]]
            else:
                pose_inputs = [pose_feats[0], pose_feats[f_i]]

            pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]

            axisangle, translation, K = pose_dec(pose_inputs, compute_intrinsic=True)
            frame_output["axisangle"] = axisangle
            frame_output["translation"] = translation
            frame_output["K"] = K

            # Invert the matrix if the frame id is negative
            frame_output["cam_T_cam"] = transformation_from_parameters(
                axisangle[:, 0], translation[:, 0], invert=(f_i < 0))

            # Add to outputs
            outputs.append(frame_output)

    return outputs

def get_rotation(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    pose_enc, pose_dec = load_models(args.model_path)
    train_loader = load_dataset(args)

    fx = []
    fy = []
    cx = []
    cy = []
    rx = []
    ry = []
    rz = []


    for batch_idx, inputs in enumerate(train_loader):
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(device)

        with torch.no_grad():
            predicted_poses = predict_pose(inputs, pose_enc, pose_dec, args)

        for frame_output in predicted_poses:
            # Convert axis angle to rotation matrix
            R = rot_from_axisangle(frame_output["axisangle"][:, 0])

            # Convert to euler angles (Modified from https://learnopencv.com/rotation-matrix-to-euler-angles/)
            sy = torch.sqrt(R[:,0,0] * R[:,0,0] +  R[:,1,0] * R[:,1,0])
            singular = sy < 1e-6

            x = torch.where(singular, torch.atan2(-R[:,1,2], R[:,1,1]), torch.atan2(R[:,2,1] , R[:,2,2]))
            y = torch.atan2(-R[:,2,0], sy)
            z = torch.where(singular, torch.zeros(R.shape[0]).to(device), torch.atan2(R[:,1,0], R[:,0,0]))

            rx.append(x.cpu())
            ry.append(y.cpu())
            rz.append(z.cpu())

            fx.append(frame_output["K"][:,0,0].cpu())
            fy.append(frame_output["K"][:,1,1].cpu())
            cx.append(frame_output["K"][:,0,2].cpu())
            cy.append(frame_output["K"][:,1,2].cpu())
    
    fx = torch.cat(fx)
    fy = torch.cat(fy)
    cx = torch.cat(cx)
    cy = torch.cat(cy)
    rx = torch.cat(rx)
    ry = torch.cat(ry)
    rz = torch.cat(rz)

    output = {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "rx": rx, "ry": ry, "rz": rz}

    return output

def plot_rotation(rotation, args):
    os.makedirs(args.save_path, exist_ok=True)

    if args.dataset in ["kitti", "kitti_odom"]:
        gt_fx = 0.58
        gt_fy = 1.92
        gt_cx = 0.5
        gt_cy = 0.5
    elif args.dataset in ["nyuv2", "nyuv2_50k"]:
        gt_fx = 0.8107155
        gt_fy = 1.0822283
        gt_cx = 0.5087226
        gt_cy = 0.528617
    elif args.dataset in ["cityscapes_preprocessed"]:
        print("TODO")
        return None
    else:
        return None

    # fx
    min_x = torch.min(rotation["ry"])
    max_x = torch.max(rotation["ry"])

    # Normalized version of Equation 3 in Depth from Video in the Wild
    r = np.linspace(min_x, max_x, 100)
    tol_pos = gt_fx + np.abs(2 * (gt_fx ** 2) / r) / args.width
    tol_neg = gt_fx - np.abs(2 * (gt_fx ** 2) / r) / args.width

    fig, ax = plt.subplots()
    ax.plot(r, tol_pos, color="red")
    ax.plot(r, tol_neg, color="red")
    ax.axhline(gt_fx, linestyle="--")
    ax.scatter(rotation["ry"], rotation["fx"], s=2)

    ax.set_ylim(tol_neg[45], tol_pos[45])

    ax.set_xlabel("Rotation Angle around the y-axis")
    ax.set_ylabel("Focal Length (x-direction)")
    fig.savefig(os.path.join(args.save_path, "fx.png"))

    # fy
    min_x = torch.min(rotation["rx"])
    max_x = torch.max(rotation["rx"])

    # Normalized version of Equation 3 in Depth from Video in the Wild
    r = np.linspace(min_x, max_x, 100)
    tol_pos = gt_fy + np.abs(2 * (gt_fy ** 2) / r) / args.height
    tol_neg = gt_fy - np.abs(2 * (gt_fy ** 2) / r) / args.height

    fig, ax = plt.subplots()
    ax.plot(r, tol_pos, color="red")
    ax.plot(r, tol_neg, color="red")
    ax.axhline(gt_fy, linestyle="--")
    ax.scatter(rotation["rx"], rotation["fy"], s=2)

    ax.set_ylim(tol_neg[45], tol_pos[45])

    ax.set_xlabel("Rotation Angle around the x-axis")
    ax.set_ylabel("Focal Length (y-direction)")
    fig.savefig(os.path.join(args.save_path, "fy.png"))

    # cx
    fig, ax = plt.subplots()
    ax.axhline(gt_cx, linestyle="--")
    ax.scatter(rotation["ry"], rotation["cx"], s=2)

    ax.set_ylim(gt_cx - 0.2, gt_cx + 0.2)

    ax.set_xlabel("Rotation Angle around the y-axis")
    ax.set_ylabel("Center (x-direction)")
    fig.savefig(os.path.join(args.save_path, "cx.png"))

    # cy
    fig, ax = plt.subplots()
    ax.axhline(gt_cy, linestyle="--")
    ax.scatter(rotation["rx"], rotation["cy"], s=2)

    ax.set_ylim(gt_cy - 0.2, gt_cy + 0.2)

    ax.set_xlabel("Rotation Angle around the x-axis")
    ax.set_ylabel("Center (y-direction)")
    fig.savefig(os.path.join(args.save_path, "cy.png"))


if __name__ == "__main__":
    args = parse_args()
    rotation = get_rotation(args)
    plot_rotation(rotation, args)

    # Save rotation to file
    torch.save(rotation, os.path.join(args.save_path, "rotation.pt"))