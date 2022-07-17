# Real-Time Video Depth Estimation

A real-time video depth estimation program based on [ManyDepth](https://github.com/nianticlabs/manydepth).

## Set Up Environment:  
```
conda create --name manydepth
conda env update --file environment.yml --prune
```  
or 
```
conda create --name manydepth
conda install -c conda-forge opencv matplotlib pytorch=1.11.0 torchvision=0.12.0 scikit-image=0.19.3 numpy=1.19.5 cudatoolkit=11.3
pip install Pillow==6.2.1 tensorboardX==1.5 tqdm==4.57.0
```
If there is an error importing tensorboardX, try `pip install protobuf==3.19.0`.


## Datasets:  

### KITTI: 
First download the raw dataset, and convert the images to jpg for faster loading time. More details can be found in [MonoDepth2](https://github.com/nianticlabs/monodepth2).
1. `wget -i <path to KITTI txt file> -P <path for saving KITTI>`
2. `find <path to KITTI> -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'`

### NYUv2
Official Website: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html

The raw dataset **(428GB)**: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip

The labelled dataset (2.8GB) with train and test split: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat

Download NYUv2 (50K Subset) and NYUv2 test set (4.4GB in total) from DenseDepth(https://github.com/ialhashim/DenseDepth)

[Indoor-SfMLearner](https://github.com/svip-lab/Indoor-SfMLearner) has a dataloader similar to monodataset.py that loads NYUv2. It is also a paper that deals with indoor depth estimation.

Extract RGB images and depth maps from raw data:
1. Download the raw dataset and extract it into folder A.
2. Download the toolbox from the official website, rename the folder as "tools" and put it in folder A.
3. Convert all RGB and depth map images to png (This command takes days to run. Command with GNU parallel like the one above for KITTI could be faster):
```
find . -name "*.ppm" -print0|xargs -I{} -0 mogrify -format png {}
find . -type f -name '*.ppm' -delete
```
4. Download `preprocess_raw.m` from https://github.com/wangq95/NYUd2-Toolkit
5. Line 15 of `preprocess_raw.m` is missing a variable name. Replace it by `for f = 1:numel(nameFolds)`.
6. Change the `savePath` and `stride` variables as you like. `stride` controls the number of output files. The default value of stride is 1, which will save all images.
7. Run `process_raw.m`: 
```
matlab -nodisplay -r "cd $(pwd); run('process_raw'); exit;"
```

Note: `kitchen_0030a/d-1315161523.972826-49330566.png` and `kitchen_0030a/d-1315161523.929615-47328411.png` might cause an error in `fill_depth_colorization.m`, so I deleted them. Some folders are empty, so they are deleted as well.

### Eigen Split
The paper with Eigen split: https://arxiv.org/pdf/1406.2283.pdf

### Scannet
Indoor video dataset.

## Results

|                       | Abs Rel | Sq Rel | RMSE  | RMSE(log) | &delta; < 1.25 | &delta; < 1.25<sup>2</sup> | &delta; < 1.25<sup>3</sup> |
| :-------------------: | :-----: | :----: | :---: | :-------: | :------------: | :------------------------: | :------------------------: |
| KITTI                 | 0.092   | 0.711  | 4.238 |  0.171    | 0.910          | 0.966                      | 0.983                      |
| KITTI (Intrinsic)     | 0.107   | 0.851  | 4.639 |  0.183    | 0.888          | 0.961                      | 0.982                      |
| NYUv2_50K             | 0.184   | 0.654  | 2.614 |  0.222    | 0.717          | 0.939                      | 0.986                      |

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

- [ ] Print loss at the end of training

## Notes

1. Two minor modifications were made following the suggestion [here](https://github.com/nianticlabs/manydepth/issues/32).

2. `frame_idxs == "s"` means the opposite stereo frame.

3. monodataset.py line 198. Always False??? What is the code for?

4. monodataset.py dummy values, why is the shape (100, 100, 3)?

5. Why is the scale 2 in line 363 of trainer.py? Because the resolution is quartered for matching (warping).

## Questions

1. NYUv2 images are unaligned??? According to: https://arxiv.org/pdf/2007.07696.pdf

2. What is the train test split??? There are so many different versions...

3. Can't compare with other papers because most of them uses the official test split, which is not a video sequence. Split the raw dataset myself, train with sequences from several scenes and evaluate with sequences from other scenes? Compare in ablation study only?

4. How do people tune hyperparameters for models that need a long training time?

2. 