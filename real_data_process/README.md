# Process Real Data 

This is the script for processing the real data.

You can download our original data for testing from [here](https://drive.google.com/drive/folders/1Fotb4KSm-aH-CAvHHPTlCVLmTFvgWRvr?usp=drive_link). Every data contains original images and colmap results, like
```
case_name/
|-- original_images/    # original capture images
    |-- IMAGE....jpg
    |-- IMAGE....jpg    
|-- images/             # downsampled images
    |-- 001.png
    |-- 002.png
|-- sparse/              # colmap results
|-- cameras.txt
|-- database.db
|-- images.txt
|-- points3D.npy
|-- points3D.txt
```

## Usage

For a new real data, we expect it to have the following structure
```
DATA_ROOT
|-- case_name/
    |-- original_images/    # original capture images
        |-- IMAGE....jpg
        |-- IMAGE....jpg    
```

First, Run
```shell
python preprocess.py case_name
```
It would downsample the original images and run colmap. Please configure colmap in advance.

After that, run
```shell
python postprocess.py case_name
```

The script would do two things: 

1. Use all sfm points to calculate the plane location and transform the world coordinate system, so that the plane would be z=0
2. Save the camera params to cameras_sphere.npz

During runing, the script needs the user to manually input x,y shift and a scale fator to ensure the object is within the unit sphere.



