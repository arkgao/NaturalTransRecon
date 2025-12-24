import numpy as np
import os.path as osp
import os        
import cv2
import argparse
from glob import glob


def downsample_images(ori_path,target_path,scale):
    img_list = sorted(glob(ori_path+'/*.jpg'))
    
    for idx,img_path in enumerate(img_list):
        img = cv2.imread(img_path)
        resolution = [round(scale*img.shape[1]),round(scale*img.shape[0])]
        if idx == 0:
            print('The output size is {}'.format(resolution))
        
        img = cv2.resize(img,resolution)
        cv2.imwrite(target_path+'/{}.png'.format(str(idx+1).zfill(3)),img)


def run_colmap(project_path):
    database_path = osp.join(project_path, "database.db")
    images_path = osp.join(project_path, "images")
    output_path = osp.join(project_path, "sparse")
    # extract features
    cmd = f"colmap feature_extractor --database_path {database_path} --image_path {images_path} --ImageReader.camera_model SIMPLE_RADIAL --ImageReader.single_camera 1"
    os.system(cmd)
    # match features
    cmd = f"colmap exhaustive_matcher --database_path {database_path}"
    os.system(cmd)
    # reconstruct sparse model
    cmd = f"mkdir -p {output_path}"
    os.system(cmd)
    cmd = f"colmap mapper --database_path {database_path} --image_path {images_path} --output_path {output_path} --Mapper.ba_refine_principal_point true"
    os.system(cmd)
    # convert to txt
    cmd = f"colmap model_converter --input_path {output_path}/0 --output_path {project_path} --output_type TXT"
    os.system(cmd)


def extract_points_array(path, savePath):
    arr = []
    with open(path, 'r') as f:
        for line in f.readlines():
            if '#' in line:
                continue
            #====== Extract tokens ===========
            tokens = line.split(' ')
            pID, P = tokens[0], tokens[1:4]
            arr.append(np.array(P, dtype=np.float32))
    arr = np.stack(arr, axis=0)
    np.save(osp.join(savePath,'points3D.npy'), arr)
            



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True)
    args = parser.parse_args()
    data_root = 'RealRawData'
    case_name = args.case
    project_path = osp.join(data_root, case_name)
    ori_img_path = osp.join(data_root, case_name, 'original_images')
    target_path = osp.join(data_root, case_name, 'images')
    downsample_scale = 0.125
    
    os.makedirs(target_path,exist_ok=True)
    downsample_images(ori_img_path,target_path,downsample_scale)


    # NOTE: Here we run colmap on the downsampled images, so it can run faster and the camera intrinsics can be directly used without adjusting. But it maybe slightly hurt the sfm quality.
    run_colmap(project_path)

    path = osp.join(project_path, 'points3D.txt')
    savePath = osp.join(project_path, 'points3D.npy')
    extract_points_array(path, savePath)