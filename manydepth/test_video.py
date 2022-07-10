import argparse
from collections import deque
import os
import multiprocessing
import queue
import time
import numpy as np
import cv2 as cv
import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torchvision import transforms

from manydepth import networks
from .test_simple import load_and_preprocess_intrinsics
from .layers import transformation_from_parameters

def parse_args():
    parser = argparse.ArgumentParser(
        description='Captures video with camera and outputs the depth map.')

    parser.add_argument('--video_path', type=str,
                        help='path to a video file',
                        required=False)
    parser.add_argument('--camera_index', type=int,
                        help='index of the camera for capturing video',
                        default=0)
    parser.add_argument('--intrinsics_json_path', type=str,
                        help='path to a json file containing a normalised 3x3 intrinsics matrix',
                        required=True)
    parser.add_argument('--model_path', type=str,
                        help='path to a folder of weights to load', required=True)
    parser.add_argument('--save_path', type=str,
                        help='path to save the depth maps as video',
                        required=False)
    parser.add_argument('--all_frames',
                        help='compute depth for all frames',
                        action="store_true")
    parser.add_argument('--depth_freq', type=int,
                        help='frequency of computing the depth. For example, depth_freq=2 means the program computes depth every 2 frames, skipping 1 frame in between.',
                        default=1)
    parser.add_argument('--no_display',
                        help='do not display the videos',
                        action="store_true")
    parser.add_argument('--mode', type=str, default='multi', choices=('multi', 'mono'),
                        help='"multi" or "mono". If set to "mono" then the network is run without '
                             'the source image, e.g. as described in Table 5 of the paper.',
                        required=False)
    return parser.parse_args()


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

    return encoder, depth_decoder, pose_enc, pose_dec, encoder_dict


def preprocess_image(image, resize_width, resize_height):
    original_width, original_height = image.shape[1], image.shape[0]
    image = cv.resize(image, (resize_width, resize_height), interpolation=cv.INTER_LANCZOS4)
    image = transforms.ToTensor()(image).unsqueeze(0)
    if torch.cuda.is_available():
        return image.cuda(), (original_height, original_width)
    return image, (original_height, original_width)


def get_depth(source_image, input_image, encoder, depth_decoder, pose_enc, pose_dec, encoder_dict, K, invK):
    """Use pretrained model to compute the depth of the input image."""

    # Preprocess images
    input_image, original_size = preprocess_image(input_image, resize_width=encoder_dict['width'], resize_height=encoder_dict['height'])

    if source_image is None:
        source_image = input_image * 0
    else:
        source_image, _ = preprocess_image(source_image, resize_width=encoder_dict['width'], resize_height=encoder_dict['height'])
    

    with torch.no_grad():
        # Estimate poses
        pose_inputs = [source_image, input_image]
        pose_inputs = [pose_enc(torch.cat(pose_inputs, 1))]
        axisangle, translation = pose_dec(pose_inputs)
        pose = transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=True)

        if args.mode == 'mono':
            pose *= 0  # zero poses are a signal to the encoder not to construct a cost volume
            source_image *= 0

        # Estimate depth
        output, lowest_cost, _ = encoder(current_image=input_image,
                                         lookup_images=source_image.unsqueeze(1),
                                         poses=pose.unsqueeze(1),
                                         K=K,
                                         invK=invK,
                                         min_depth_bin=encoder_dict['min_depth_bin'],
                                         max_depth_bin=encoder_dict['max_depth_bin'])

        output = depth_decoder(output)

        sigmoid_output = output[("disp", 0)]
        sigmoid_output_resized = torch.nn.functional.interpolate(
            sigmoid_output, original_size, mode="bilinear", align_corners=False)
        sigmoid_output_resized = sigmoid_output_resized.cpu().numpy()[:, 0]

        # Generate colormapped depth image
        toplot = sigmoid_output_resized.squeeze()
        normalizer = mpl.colors.Normalize(vmin=toplot.min(), vmax=np.percentile(toplot, 95))
        mapper = cm.ScalarMappable(norm=normalizer, cmap='Greys_r')
        depth_colormap = (mapper.to_rgba(toplot)[:, :, :3] * 255).astype(np.uint8)

    return depth_colormap


def read_frame(frame_queue, fps_queue, camera_index=0, video_path=None, display=True):
    use_camera = video_path is None

    # Turn on camera or read video
    if use_camera:
        cap = cv.VideoCapture(camera_index)
    else:
        cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot turn on camera." if use_camera else "Cannot read video.")
        exit()

    fps = int(cap.get(5))
    fps_queue.put(fps)

    # Capture and display video until user press "q"
    while True:
        ret, current_frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Video ended / Cannot correctly read frame.")
            break

        # Put frame to queue
        frame_queue.put(current_frame)

        if display:
            cv.imshow("Input Video", cv.resize(current_frame, (current_frame.shape[1] // 2, current_frame.shape[0] // 2)))
        
        # Stop when user press "q"
        if cv.waitKey(1) == ord('q'):
            break
    
    # Release all resources used by VideoCapture
    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


def test_video(args):
    """Capture video using a connected camera and output the corresponding depth map"""
    if args.save_path is not None:
        assert args.save_path[-4:] == ".avi", \
            "save_path must end with '.avi'"
    
    display = False if args.no_display else True

    # Load pretrained models
    encoder, depth_decoder, pose_enc, pose_dec, encoder_dict = load_models(args.model_path)

    # Load and preprocess intrinsics
    K, invK = load_and_preprocess_intrinsics(args.intrinsics_json_path,
                                             resize_width=encoder_dict['width'],
                                             resize_height=encoder_dict['height'])                                         

    # Initialize variables
    depth_map_deque = deque()
    previous_frame = None
    current_frame = None
    total_frames = 0 # This allows depth computation of the first frame

    # Create a separate process to read frames from video, so a smooth video can be displayed and the main process can achieve a higher FPS.
    frame_queue = multiprocessing.Manager().Queue() # multiprocessing.Queue() does not work. See here: https://stackoverflow.com/questions/47085458/why-is-multiprocessing-queue-get-so-slow
    fps_queue = multiprocessing.Queue() # Only for getting fps, so multiprocessing.Queue() works here.

    read_frame_process = multiprocessing.Process(target=read_frame,args=(frame_queue, fps_queue, args.camera_index, args.video_path, display))
    read_frame_process.start()

    # Get FPS of the input video
    while True:
        try: 
            original_fps = fps_queue.get(False)
            break
        except queue.Empty:
            continue
    print(f"Input Video FPS: {original_fps}")
    
    start_time = time.time()

    # Compute depth of the input video
    while True:
        if not read_frame_process.is_alive() and frame_queue.empty(): # Finished processing all frames in frame_queue
            break
        
        if frame_queue.empty(): # Wait for read_frames() to put new frames in frame_queue
            continue
        
        # Get the target frames
        while not frame_queue.empty():
            previous_frame = current_frame
            current_frame = frame_queue.get()
            total_frames += 1

            if args.all_frames:
                break
        
        # Skipping frames
        if args.depth_freq and total_frames % args.depth_freq != 0:
            continue

        # Process frame to get depth
        depth = get_depth(previous_frame, current_frame, encoder, depth_decoder, pose_enc, pose_dec, encoder_dict, K, invK)

        depth_map_deque.append(depth)
        
        if display:
            cv.imshow("Input Frame", cv.resize(current_frame, (current_frame.shape[1] // 2, current_frame.shape[0] // 2)))
            cv.imshow("Depth Map", cv.resize(depth, (depth.shape[1] // 2, depth.shape[0] // 2)))
        
        # Stop when user press "q"
        if cv.waitKey(1) == ord('q'):
            break
    
    time_elapsed = time.time() - start_time
    num_processed_frame = len(depth_map_deque)
    print(f"Number of frames in the video: {total_frames}")
    print(f"Number of frames processed: {num_processed_frame}")
    print(f"Depth Computation FPS: {num_processed_frame / time_elapsed}")
    
    cv.destroyAllWindows()
    cv.waitKey(1)

    read_frame_process.terminate()
    read_frame_process.join()

    # Save depth map as video
    if args.save_path and depth_map_deque:
        print("Saving depth video...")
        new_fps = original_fps * len(depth_map_deque) / total_frames
        frame_width, frame_height = depth.shape[1], depth.shape[0]
        depth_video_writer = cv.VideoWriter(args.save_path,cv.VideoWriter_fourcc('M','J','P','G'), new_fps, (frame_width,frame_height))  
        while depth_map_deque:
            depth_video_writer.write(depth_map_deque.popleft())
        print(f"Saved depth video to {args.save_path}.")
        print(f"Output Video FPS: {new_fps}")

if __name__ == '__main__':
    args = parse_args()
    test_video(args)