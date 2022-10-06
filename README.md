# Real-Time Video Depth Estimation

A real-time video depth estimation program based on [ManyDepth](https://github.com/nianticlabs/manydepth) and [Depth from videos in the wild](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild).

## Set Up Environment:  
```
conda create --name manydepth
conda activate manydepth
conda env update --file environment.yml --prune
```  
or 
```
conda create --name manydepth
conda activate manydepth
conda install -c conda-forge opencv matplotlib pytorch=1.11.0 torchvision=0.12.0 scikit-image=0.19.3 numpy=1.19.5 cudatoolkit=11.3
pip install Pillow==6.2.1 tensorboardX==1.5 tqdm==4.57.0
```
If there is an error importing tensorboardX, try `pip install protobuf==3.19.0`.

## How to Run
### **To generate a depth map video of your own video**
```
python -m manydepth.test_video --model_path <Path to pretrained model> --save_path <Path to save output video (.avi)> --video_path <Path to input video>
```
Use `--intrinsics_json_path <Path to intrinsic JSON file>` to provide ground truth camera intrinsic for prediction.  
Use `--no_display` to disable video display.

### **To generate depth map using camera input in real time**
```
python -m manydepth.test_video --model_path <Path to pretrained model> --real_time
```
Use `--camera_index <index>` to choose camera.

### **To train a model**
Follow the next section to download and preprocess the datasets, then run the command below.
```
python -u -m manydepth.train --data_path <Path to dataset> --log_dir <Path to log> --model_name <Model name> --png --num_workers 12
```
Example for KITTI dataset:
```
python -m manydepth.train --data_path data_sets/KITTI --log_dir manydepth/log --model_name kitti --png --num_workers 12
```
Example for NYUv2 (50K) subset:
```
python -m manydepth.train --data_path data_sets/NYUv2_50K --dataset nyuv2_50k --split nyuv2_50k --log_dir manydepth/log --model_name nyu50k --num_workers 12 --max_depth 10.0 --height 288 --width 384
```

Example for Transfer Learning:
```
python -m manydepth.train --data_path data_sets/NYUv2_50K --dataset nyuv2_50k --split nyuv2_50k --log_dir manydepth/log --model_name nyu50k_from_kitti_intrinsic --num_workers 12 --max_depth 10.0 --height 192 --width 640 --num_epochs 10 --freeze_teacher_epoch 5 --load_weights_folder manydepth/pretrained_weights/kitti_intrinsic/final
```

--resnet_path: use pre-downloaded resnet models  
--no_compute_intrinsic: model will not learn to predict camera intrinsic

### **To evaluate a model**
KITTI dataset:  
Remember to extract ground truth depth before evaluation.
```
python -m manydepth.export_gt_depth --data_path <Path to KITTI> --split eigen
```
```
python -m manydepth.evaluate_depth --data_path <Path to KITTI> --load_weights_folder <Path to trained model> --eval_mono --png
```
NYUv2:  
Again, extract the ground truth depth before evaluation.
```
python -m manydepth.export_nyuv2_gt_depth --data_path <Path to NYU> --split nyuv2
```
```
python -m manydepth.evaluate_depth --data_path <Path to NYU> --load_weights_folder <Path to trained model> --eval_mono --png --eval_split nyuv2
```

## Datasets

### KITTI: 
First download the raw dataset, and convert the images to jpg for faster loading time. More details can be found in [MonoDepth2](https://github.com/nianticlabs/monodepth2).
1. `wget -i <path to KITTI txt file> -P <path for saving KITTI>`
2. `find <path to KITTI> -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'`

### NYUv2
Download NYUv2 (50K Subset) and NYUv2 test set (4.4GB in total) from DenseDepth(https://github.com/ialhashim/DenseDepth)

If you need to use the raw dataset, you may follow the following links and instructions below.

Official Website: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

The raw dataset **(428GB)**: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip

[Indoor-SfMLearner](https://github.com/svip-lab/Indoor-SfMLearner) has a dataloader similar to monodataset.py that loads NYUv2. It is also a paper that deals with indoor depth estimation.

Extract RGB images and depth maps from raw data:
1. Download the raw dataset and extract it into folder A.
2. Download the toolbox from the official website, rename the folder as "tools" and put it in folder A.
3. Convert all RGB and depth map images to png (This command takes days to run. Command with GNU parallel like the one above for KITTI could be faster):
```
find . -name "*.ppm" -print0|xargs -I{} -0 mogrify -format png {}
find . -type f -name '*.ppm' -delete

find . -name "*.pgm" -print0|xargs -I{} -0 mogrify -format png {}
find . -type f -name '*.pgm' -delete
```
4. Download `preprocess_raw.m` from https://github.com/wangq95/NYUd2-Toolkit
5. Line 15 of `preprocess_raw.m` is missing a variable name. Replace it by `for f = 1:numel(nameFolds)`.
6. Change the `savePath` and `stride` variables as you like. `stride` controls the number of output files. The default value of stride is 1, which will save all images.
7. Run `process_raw.m`: 
```
matlab -nodisplay -r "cd $(pwd); run('process_raw'); exit;"
```

Note: `kitchen_0030a/d-1315161523.972826-49330566.png` and `kitchen_0030a/d-1315161523.929615-47328411.png` might cause an error in `fill_depth_colorization.m`, so I deleted them. Some folders are empty, so they are deleted as well.

### Other metrics

1. Frame per second (FPS)
2. Temporal Consistency 
    - Absolute relative temporal error (ARTE) in [*Do not Forget the Past: Recurrent Depth Estimation from Monocular Video*](https://arxiv.org/abs/2001.02613).
    - Instability and drift in [*Consistent Video Depth Estimation*](https://roxanneluo.github.io/Consistent-Video-Depth-Estimation/).
    - Temporal consistency loss in [*Enforcing Temporal Consistency in Video Depth Estimation*](https://openaccess.thecvf.com/content/ICCV2021W/PBDL/papers/Li_Enforcing_Temporal_Consistency_in_Video_Depth_Estimation_ICCVW_2021_paper.pdf) or [Exploiting temporal consistency for real-time video depth estimation](https://arxiv.org/abs/1908.03706).
3. For camera intrinsics, follow [Depth from Videos in the Wild:
Unsupervised Monocular Depth Learning from Unknown Cameras](https://arxiv.org/abs/1904.04998).

## To do

- [x] Run the codes
- [x] Real Time Video Prediction
- [x] Video Prediction (All frames)
- [x] Train on KITTI to reproduce results
- [x] Create Dataloader for NYUv2
- [x] Train on NYUv2 and other indoor datasets (Possible challenges can be found here: https://github.com/nianticlabs/manydepth/issues/35)
- [x] Learn camera intrinsics
- [ ] Improve accuracy
- [ ] Compute additional metrics, e.g. temporal consistency

### Other minor changes

- [x] Print loss at the end of training
- [ ] Adjust intrinsics during inference?

## Notes

1. Two minor modifications were made following the suggestion [here](https://github.com/nianticlabs/manydepth/issues/32).

2. `frame_idxs == "s"` means the opposite stereo frame.

3. monodataset.py line 198. Always False??? What is the code for?

4. Scale 2 in line 363 of trainer.py because the resolution is quartered for matching (warping).

## References

1. ManyDepth:  
https://github.com/nianticlabs/manydepth

2. Official implementation of *Depth from videos in the wild*:  
https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild

3. Pytorch implementation of *Depth from videos in the wild*:  
https://github.com/bolianchen/pytorch_depth_from_videos_in_the_wild

4. Indoor-SfMLearner:  
https://github.com/svip-lab/Indoor-SfMLearner