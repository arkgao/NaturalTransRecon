import numpy as np
import os.path as osp
import os
from utils import quate2rotmat, fit_plane_RANSAC, cal_plane_transform_matrix, normalize
import argparse


def load_camera_intrin(camerainfo_path):
    camera_dict = {}
    # real the camera intrisict matrix
    c = 0
    with open(camerainfo_path, 'r') as camParams:
        for p in camParams.readlines():
            c += 1
            if c <= 3: # skip comments
                continue
            else:
                line = p.strip().split(' ')
                imgW, imgH = int(line[2]), int(line[3])
                f = float(line[4])
                cxp, cyp = int(float(line[5])), int(float(line[6]))
                intri_mat = np.eye(3)
                intri_mat[0,0] = f
                intri_mat[1,1] = f
                intri_mat[0,2] = cxp
                intri_mat[1,2] = cyp
                break
    camera_dict['intrisict_mat'] = intri_mat
    return camera_dict

def load_camera_extrin(imginfo_path, global_extric_mat):
    # load the camera extrinsic params
    # and adjust them with extra transformation
    # so the plane would be z=0
    
    c = 0
    with open(imginfo_path, 'r') as camPoses:
        for cam in camPoses.readlines():
            c += 1
            if c <= 3: # skip comments
                continue
            elif c == 4:
                numImg = int(cam.strip().split(',')[0].split(':')[1])
                print('Number of images:', numImg)
                
                # The current camera pose
                rot_mat_array = np.zeros([numImg,3,3])
                t_array = np.zeros([numImg,3,1])
                origin_array = np.ones([numImg,3])
                dir_array = np.ones([numImg,3])
                
            else:
                if c % 2 == 1:
                    line = cam.strip().split(' ')
                    img_idx = int(line[9][0:-4])-1
                    ori_rot_mat = quate2rotmat(line[1:5])
                    ori_t = np.array([float(line[5]), float(line[6]), float(line[7])])
                    
                    ori_extri_mat = np.concatenate((ori_rot_mat, ori_t.reshape(3,1)), axis=1)
                    ori_extri_mat = np.concatenate([ori_extri_mat, np.array([[0,0,0,1]])], axis=0)
                    extri_mat = np.dot(ori_extri_mat,global_extric_mat)[:3,:]
                    
                    rot_mat = extri_mat[:3,:3]
                    t = extri_mat[:3,3]
                    
                    origin = -np.dot(np.transpose(rot_mat),t.reshape(3,1)).reshape(3)
                    dir = np.dot(np.transpose(rot_mat),np.array([0,0,1]).reshape(3,1)).reshape(3)
                    
                    dir_array[img_idx] = dir
                    origin_array[img_idx] = origin
                    rot_mat_array[img_idx] = rot_mat
                    t_array[img_idx] = t.reshape(3,1)
                    
    return dir_array, origin_array, rot_mat_array, t_array

def save_camera_params(origin_array, rot_mat_array, t_array, camera_dict, save_path):

    # print the distrubution of the camera center
    print('The distribution of the camera center:')
    print('x: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,0]), np.max(origin_array[:,0]), np.mean(origin_array[:,0])))
    print('y: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,1]), np.max(origin_array[:,1]), np.mean(origin_array[:,1])))
    print('z: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,2]), np.max(origin_array[:,2]), np.mean(origin_array[:,2])))

    # shift and scale to the unit sphere
    x_shift = -(np.max(origin_array[:,0]) + np.min(origin_array[:,0]))/2
    y_shift = -(np.max(origin_array[:,1]) + np.min(origin_array[:,1]))/2

    print('please set the extra shift vector -- x_shift, y_shift')
    line = input().strip().split(' ')
    extra_x_shift = float(line[0])
    extra_y_shift = float(line[1])


    shift_vec = np.array([x_shift+extra_x_shift,y_shift+extra_y_shift,0]).reshape(3,1)


    print('please set the scale')
    scale = float(input())
    scale_mat = scale*np.eye(4)
    scale_mat[3,3] = 1
    check_list = []
    intri_mat = camera_dict['intrisict_mat']
    numImg = origin_array.shape[0]

    for idx in range(numImg):
        extri_mat = np.concatenate([rot_mat_array[idx], t_array[idx]], axis=1)
        extri_mat = np.concatenate([extri_mat, np.array([[0,0,0,1]])], axis=0)
        extri_mat[0:3,3] = extri_mat[0:3,3] + -np.dot(extri_mat[0:3,0:3],shift_vec).reshape(3)
        # extri_mat = np.dot(extri_mat,scale_mat)
        rot_mat = extri_mat[0:3,0:3]
        t = extri_mat[0:3,3]
        origin = -np.dot(np.transpose(rot_mat),t.reshape(3,1)).reshape(3)
        origin_array[idx] = origin
        
        camera_dict['rot_mat_{}'.format(idx)] = rot_mat                
        camera_dict['origin_{}'.format(idx)] = origin
        camera_dict['world_mat_{}'.format(idx)] = np.dot(intri_mat,extri_mat[0:3,:])    
        camera_dict['scale_mat_{}'.format(idx)] = scale_mat
        z_axis = np.dot(np.transpose(rot_mat),np.array([0,0,1]).reshape(3,1)).reshape(3)
        check = np.sum(z_axis*normalize(-origin))
        check_list.append(check>0)

    print(np.sum(np.array(check_list)))
    print('The distribution of the final camera center:')
    print('x: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,0]), np.max(origin_array[:,0]), np.mean(origin_array[:,0])))
    print('y: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,1]), np.max(origin_array[:,1]), np.mean(origin_array[:,1])))
    print('z: min:{}, max:{}, mean:{}'.format(np.min(origin_array[:,2]), np.max(origin_array[:,2]), np.mean(origin_array[:,2])))

    np.savez(os.path.join(save_path,'cameras_sphere.npz'), **camera_dict)

    object_dict = {}
    object_dict['scale_mat_0'] = scale_mat
    np.savez(os.path.join(save_path,'object_sphere.npz'), **object_dict)

    with open(os.path.join(save_path,'camera_configure.txt'),'w') as f:
        f.write('scale: {}\n'.format(scale))
        f.write('extra shift: x:{} y:{}\n'.format(extra_x_shift,extra_y_shift))

def calculate_plane_location(project_path):
    points_path = osp.join(project_path, 'points3D.npy')
    points = np.load(points_path)
    A,B,C,D = fit_plane_RANSAC(points)
    if C<0:
        A,B,C,D = -A,-B,-C,-D
    return A,B,C,D

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--flip_z', type=bool, default=True)
    args = parser.parse_args()
    case_name = args.case
    data_root = 'RealRawData'

    # Despite we can automatically locate the plane location and transform it to z=0, we still need to specify the z-axis direction so that the object on the side where z>0.
    # This can be automatically achieved by determining the sign of the camera's z coordinate.
    # But we simply achieve it through manual flag.
    FLIP_FLAG = args.flip_z # whether flip the z axis

    
    imginfo_path = osp.join(data_root, case_name, 'images.txt')
    camerainfo_path = osp.join(data_root, case_name, 'cameras.txt')
    point3d_path = osp.join(data_root, case_name, 'points3D.npy')
    save_path = osp.join(data_root,case_name)
    
    # fit the plane location and calculate a extra transformation matrix
    points = np.load(point3d_path)
    A,B,C,D = fit_plane_RANSAC(points)
    if C<0:
        A,B,C,D = -A,-B,-C,-D
    #! warning the inverse mode is true, and that's is correct for the following calculation
    global_extric_mat = cal_plane_transform_matrix(A,B,C,D,inverse_mode=True)

    if FLIP_FLAG:
        z_flip_mat = np.eye(4)
        z_flip_mat[1,1] = -1
        z_flip_mat[2,2] = -1
        global_extric_mat = np.dot(global_extric_mat, z_flip_mat)


    camera_dict = load_camera_intrin(camerainfo_path)
    dir_array, origin_array, rot_mat_array, t_array = load_camera_extrin(imginfo_path,global_extric_mat)
    save_camera_params(origin_array,rot_mat_array,t_array,camera_dict,save_path)