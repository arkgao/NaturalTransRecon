# Transparent Object Reconstruction via Implicit Differentiable Refraction Rendering
This is the official implementation of the paper "Transparent Object Reconstruction via Implicit Differentiable Refraction Rendering". [SIGGRAPH Asia 2023]

![](asset/teaser.jpg)

## [Project page] | [Paper] TODO
# Usage

## Setup
1. Install torch and torchvision according to your environment. For reference, we use torch==2.0.1+cu118.
2. Install other packages.
    ```shell
    git clone https://github.com/arkgao/NaturalTransRecon.git
    cd NaturalTransRecon
    pip install -r requirements.txt
    ```

## Download Datasets
The dataset and our results can be downloaded from [here](https://drive.google.com/drive/folders/1Fotb4KSm-aH-CAvHHPTlCVLmTFvgWRvr?usp=drive_link)

The data structure should be like:
```
NaturalTransRecon/
|-- data/
    |-- case_name_1/
        |-- image/              # multi-view images
            |-- 000.png     
            |-- 001.png
                ...
        |-- mask/               # multi-view masks
        |-- normal/             # multi-view normals
        |-- cameras_sphere.npz  # camera params
        |-- object_sphere.npz   # camera params
        |-- gt.ply              # ground truth mesh
    |-- case_name_2/
```
Our method only use image to reconstruct transparent objects, and the mask and normal in synthetic data are only used to validate results. The synthetic data is rendered with the shape from [Wu et al.](https://vcc.tech/research/2018/FRT) and [DRT](https://vcc.tech/research/2020/DRT). Thanks for their open-source data. For real data, there is no mask and normal. We scan the painted real objects to get their GT shapes, while the pigment may introduce error and noise on ground truth.


For convenience, the plane position in all data is z = 0. The object is located on the side where z > 0 and is within the unit sphere. Please meet this requirement if you want to test custom data. You can see below how to process the real data to meet this requirement.


Then, you can direcly run bash script
```shell
bash run.sh case_name
```
or run python script step by step as follows:
## Stage1: export object silhouettes and plane texture
```shell
# run neus without considering the refraction
python exp_runner.py --case CASE_NAME

# export the multi-view object masks
python export_mask.py --case CASE_NAME  # add --val_error to calculate the error for syn data

# export the plane texture
python export_texture.py --case CASE_NAME 
```
The results would be saved in exp folder, and it should be like:
```
NaturalTransRecon/
|-- exp/
    |-- case_name/
        |-- export_mask/
            |-- margin/
            |-- mask/
        |-- stage1
            |-- export_texture/
                ...
```


## Stage2: reconstruct the transparent object
```shell
# initialize the object shape with masks
python init_shape.py --case case_name

# optimize shape through refractive rendering
python optim_transparent.py --case case_name    # add --val_error to calculate the error for syn data
#! For real data, there is misalignment and scalar ambiguity between GT and reconstruction
# Do not use this way to calculate error for real data

```
The final mesh is stored in exp/case_name/optim_trans/meshes/00300000.ply
```
NaturalTransRecon/
|-- exp/
    |-- case_name/
        |-- export_mask/
        |-- init_shape/
        |-- optim_trans/
        |-- stage1/
```

# Pre-process on Real Data
Note that all planes in synthetic data are set as z=0, so that we can conviently deal with it in the reconstruction.
For real data, we use a script to locate the plane and transfer it to z=0. Then it can be compatible with our program. [Here is the guidance for processing real data](real_data_process/README.md)

# Our results
For reference and comparasion, we provide all our results [here](https://drive.google.com/drive/folders/1Fotb4KSm-aH-CAvHHPTlCVLmTFvgWRvr?usp=drive_link).

# Acknowledgements
This repository is heavily based on NeuS and Geo-NeuS. We thank all the authors for their excellent work and sharing great codes. And we also very grateful for the open-source mesh from [Wu et al.](https://vcc.tech/research/2018/FRT) and [DRT](https://vcc.tech/research/2020/DRT).

# Citation
```
@inproceedings{10.1145/3610548.3618236,
author = {Gao, Fangzhou and Zhang, Lianghao and Wang, Li and Cheng, Jiamin and Zhang, Jiawan},
title = {Transparent Object Reconstruction via Implicit Differentiable Refraction Rendering},
year = {2023},
url = {https://doi.org/10.1145/3610548.3618236},
doi = {10.1145/3610548.3618236},
booktitle = {SIGGRAPH Asia 2023 Conference Papers},
articleno = {57},
numpages = {11},
}
```