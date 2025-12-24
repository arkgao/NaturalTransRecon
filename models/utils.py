import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import mcubes
import trimesh 
import tqdm
import kaolin as kal


# approximatly calculate the intersection with the unit sphere
# sample more range and more friendly to those beyond unit sphere
def near_far_from_sphere(rays_o, rays_d):
    a = torch.sum(rays_d**2, dim=-1, keepdim=True)
    b = 2.0 * torch.sum(rays_o * rays_d, dim=-1, keepdim=True)
    mid = 0.5 * (-b) / a
    near = mid - 1.0
    far = mid + 1.0
    near = near.clamp(min=1e-4)
    far = torch.max(far,near+1.0)
    return near, far

def get_half_intersection(rays_o, rays_d, r=1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
    
    origin_vector = -rays_o
    ray_dot = (rays_d * origin_vector).sum(dim=1,keepdim=True)

    under_sqrt = r ** 2 - (origin_vector.norm(2,1, keepdim=True) ** 2 - ray_dot ** 2)

    mask_intersect = (under_sqrt > 1e-3) & (abs(rays_d[:,2:3])>1e-4)
    
    near = torch.zeros_like(rays_o)[:,0:1]
    far  = 2*torch.ones_like(rays_d)[:,0:1]
    near[mask_intersect] = ray_dot[mask_intersect] - torch.sqrt(under_sqrt[mask_intersect])
    near = near.clamp(min=1e-4)
    sphere_far = ray_dot[mask_intersect] + torch.sqrt(under_sqrt[mask_intersect])
    circle_far = -rays_o[mask_intersect[:,0],2]/(rays_d[mask_intersect[:,0],2]+1e-4)
    far[mask_intersect] = torch.min(sphere_far.clamp(min=1e-4),circle_far.clamp(min=1e-4))
    
    far[~ mask_intersect] = near[~ mask_intersect]+1e-4
    
    return mask_intersect.squeeze(1), near, far

# strictly calculate the intersection with the unit sphere
# more precise
def get_sphere_intersection(rays_o, rays_d, r = 1.0):
    # Input: n_images x 4 x 4 ; n_images x n_rays x 3
    # Output: n_images * n_rays x 2 (close and far) ; n_images * n_rays
    
    origin_vector = -rays_o
    ray_dot = (rays_d * origin_vector).sum(dim=1,keepdim=True)

    under_sqrt = r ** 2 - (origin_vector.norm(2,1, keepdim=True) ** 2 - ray_dot ** 2)

    mask_intersect = under_sqrt > 1e-3
    
    near = torch.zeros_like(rays_o)[:,0:1]
    far  = 2*torch.ones_like(rays_d)[:,0:1]
    near[mask_intersect] = ray_dot[mask_intersect] - torch.sqrt(under_sqrt[mask_intersect])
    near = near.clamp(min=1e-4)
    far[mask_intersect] = ray_dot[mask_intersect] + torch.sqrt(under_sqrt[mask_intersect])

    far[far<0] = 2
    
    return mask_intersect.squeeze(1), near, far

def mesh_to_point_tensor(mesh:trimesh.Trimesh,device='cuda'):
    return torch.tensor(mesh.vertices,device=device)

def normalize_point_cloud(points:torch.Tensor):
    low_bound,_ = torch.min(points,dim=0)
    high_bound,_ = torch.max(points,dim=0)
    bound = torch.stack([low_bound,high_bound],dim=0)
    length = bound[1,:] - bound[0,:]
    dialg = torch.sqrt(length[0]**2+length[1]**2+length[2]**2)
    scale = 1 / dialg
    points = points*scale
    return points, scale

def chamfer_distance_kaolin(pred_point:torch.Tensor,gt_point:torch.Tensor):
    gt_point,scale = normalize_point_cloud(gt_point)
    pred_point = pred_point*scale
    
    if len(gt_point.shape) == 2:
        gt_point = gt_point.unsqueeze(0)
    if len(pred_point.shape) == 2:
        pred_point = pred_point.unsqueeze(0)
    
    return kal.metrics.pointcloud.chamfer_distance(pred_point,gt_point).item()


def plt_show(x,y=None,filename='debug', debug_path='./debug'):
    os.makedirs(debug_path,exist_ok=True)
    if y is not None:
        x = x.squeeze().detach().cpu().numpy()
        y = y.squeeze().detach().cpu().numpy()
    else:
        y = x.squeeze().detach().cpu().numpy()
        x = np.arange(0,y.shape[0])
    
    plt.plot(x,y,'-o')
    plt.savefig(os.path.join(debug_path,'{}.png'.format(filename)))
    plt.close()

def dot(a:torch.Tensor, b:torch.Tensor, keepdim = True):   
    c = a[:,0] * b[:,0] + a[:,1] * b[:,1] + a[:,2] * b[:,2]
    if keepdim:
        c = c.view(-1,1)
    return c


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs,indexing='ij')
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def enlarge_bounding_box(bounds, rate=1.2):
    """
        enlarge the bounding box 
    Args:
        bounds (_type_): [2,3] [[x_min,y_min,z_min],[x_max,y_max,z_max]]
    """
    center = (bounds[0,:]+bounds[1,:]) / 2
    rad = (bounds[1,:] - bounds[0,:]) / 2
    rad = rate*rad
    bounds = np.vstack([center-rad, center+rad])
    return bounds

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def get_texture(renderer, resolution=512):
    z_val = 0.01
    batch_size = 128
    Xsample, Ysample = torch.meshgrid(torch.linspace(-1,1,resolution),
                                          torch.linspace(-1,1,resolution),indexing='xy')
    location = torch.stack([Xsample,Ysample,z_val*torch.ones_like(Xsample)],dim=-1)
    rays_o = location.reshape(-1,3)
    rays_d = torch.ones_like(rays_o) * torch.tensor([[0,0,-1.0]],device='cuda')
        
    rays_o = rays_o.reshape(-1, 3).split(batch_size)
    rays_d = rays_d.reshape(-1, 3).split(batch_size)
        
    color_list = []
    with torch.no_grad():
        for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
            render_output = renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            sample_dir=None
                                            )
            color_list.append(render_output['color'].detach().cpu().numpy())

    color = np.concatenate(color_list,axis=0).reshape(resolution,resolution,3).astype(np.float32)
        
    return color

class Logger(object):
    def __init__(self, path):
        self.log_file = open(os.path.join(path, 'log.txt'), 'w')
        self.output_path = path
        
    def printandwrite(self, strs):
        self.log_file.write('%s\n' % strs)
        self.log_file.flush()
        print('%s\n' % strs)