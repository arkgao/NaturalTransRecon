import numpy as np


def normalize(array):
    return array/np.linalg.norm(array)

def quate2rotmat(Q):
    # q = a + bi + cj + dk
    a = float(Q[0])
    b = float(Q[1])
    c = float(Q[2])
    d = float(Q[3])

    R = np.array([[2*a**2-1+2*b**2, 2*b*c+2*a*d,     2*b*d-2*a*c],
                  [2*b*c-2*a*d,     2*a**2-1+2*c**2, 2*c*d+2*a*b],
                  [2*b*d+2*a*c,     2*c*d-2*a*b,     2*a**2-1+2*d**2]])
    return np.transpose(R)

def fit_plane_RANSAC(points, inlier_threshold=0.1, max_iterations=100):
    """
        Given 3D points, use RANSAC to fit a plane
    return:
    A, B, C, D -- for the coefficient in the plane equation Ax + By + Cz + D = 0。
    """

    n = points.shape[0]
    best_score = 0
    best_model = None

    for i in range(max_iterations):
        # random choose 3 points
        indices = np.random.choice(n, size=3, replace=False)
        p1, p2, p3 = points[indices]

        # calculate normal
        normal = np.cross(p2 - p1, p3 - p1)
        normal /= np.linalg.norm(normal)

        # calculate offset
        offset = -np.dot(normal, p1)

        distances = np.abs(np.dot(points, normal) + offset)
        inliers = (distances < inlier_threshold).sum()

        if inliers > best_score:
            best_score = inliers
            best_model = (normal, offset)

    normal, offset = best_model
    A, B, C = normal

    distances = np.abs(np.dot(points, normal) + offset)
    inliers = points[distances < inlier_threshold]
    D = -np.dot(normal, inliers.mean(axis=0))

    return A, B, C, D


def euler_to_rotation_matrix(alpha, beta, gamma):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]
                   ])

    R_y = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]
                   ])

    R_z = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]
                   ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

def cal_rotation_matrix(v):
    # this function calculate the rotation matrix that can transform
    # the vector v to (0,0,1)
    a,b,c = v
    alpha = np.arctan(b/c)
    beta = np.arctan(-a/(b*np.sin(alpha)+c*np.cos(alpha)))
    rot_mat = euler_to_rotation_matrix(alpha, beta, 0)
    return rot_mat


def cal_plane_transform_matrix(A,B,C,D,inverse_mode=False):
    # when inverse mode is Falase
    # this function calculate transform matrix that can transform 
    # a point on the plane express as (Ax+By+Cz+D=0) to the plane z=0
    # for a point p on the plane, the transform matrix F will 
    # Fp = [a,b,0,1]T
    ori_normal = normalize(np.array([A,B,C]))
    rot_mat = cal_rotation_matrix(ori_normal)
    
    if inverse_mode:
        rot_mat = rot_mat.transpose()
        trans = np.array([0,0,-D/C]).reshape(3,1)
    else:
        trans = np.array([0,0,D/C*rot_mat[2,2]]).reshape(3,1)
    transform_matrix = np.concatenate([rot_mat,trans],axis=1)
    transform_matrix = np.concatenate([transform_matrix,np.array([0,0,0,1]).reshape(1,4)],axis=0)
    
    
    return transform_matrix

