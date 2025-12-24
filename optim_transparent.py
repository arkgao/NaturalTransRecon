import os
import logging
import argparse
import numpy as np
import cv2 as cv
import random
import trimesh
import time
import torch
import torch.nn.functional as F
from glob import glob
from torch.utils.tensorboard import SummaryWriter
from shutil import copyfile
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import SDFNetwork, SingleVarianceNetwork
from models.TransRender import TransRender
from models.utils import enlarge_bounding_box,chamfer_distance_kaolin
from models.RayTracer import RayTracer
from models.renderer import TextureRenderer
from matplotlib import cm


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def colormap(diff,thres):
    diff_norm = np.clip(diff, 0, thres) / thres
    diff_cm = cm.jet(diff_norm)[:,:, :3]
    return diff_cm[:,:,::-1]

class Exper:
    def __init__(self, conf_path, case='CASE_NAME', exp_name='optim_trans', val_error = False):
        self.device = torch.device('cuda')
        self.exp_name = exp_name
        
        
        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()
        
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        

        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])

            
        # Networks
        self.obj_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.obj_deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        
        # tracer for transparent object
        self.tracer = RayTracer(self.obj_network, self.obj_deviation_network, 
                           obj_box = self.dataset.bounding_box, **self.conf['renderer.tracer'])


        self.scene_render = TextureRenderer()
        self.renderer = TransRender(self.tracer,
                                    self.scene_render,
                                    **self.conf['renderer.trans_renderer'])
        # Load trained model in stage1 and initialized object model
        self.reload_trained_model()

        # Coarse to fine
        self.init_sigma = self.conf.get_float('optim_transparent.init_sigma')
        self.coarse_level = self.conf.get_int('optim_transparent.coarse_level')

        # Load estimated mask and margin in stage1
        if self.conf.get_string('mask_dir', default = None):
            mask_dir = self.conf.get_string('mask_dir')
        else:    
            mask_dir = os.path.join(self.base_exp_dir, 'export_mask', 'mask')
        self.dataset.load_mask(mask_dir)
        self.dataset.load_margin(os.path.join(self.base_exp_dir, 'export_mask', 'margin'))

        # Load the initialized object bounding box
        self.dataset.load_pred_bounding_box(os.path.join(self.base_exp_dir,'init_shape','bounding_box.npy'))
        self.bounds = np.load(os.path.join(self.base_exp_dir,'init_shape','bounding_box.npy'))        
        
        # Weights
        self.igr_weight = self.conf.get_float('optim_transparent.igr_weight')
        self.mask_weight = self.conf.get_float('optim_transparent.mask_weight')
        self.color_weight = self.conf.get_float('optim_transparent.color_weight', default=1.0)
        
        # Train params 
        self.warm_up_end = 0
        self.learning_rate = self.conf.get_float('optim_transparent.learning_rate')
        self.learning_rate_alpha = self.conf.get_float('optim_transparent.learning_rate_alpha')
        self.end_iter = self.conf.get_int('optim_transparent.end_iter')
        self.color_batch_size = self.conf.get_int('optim_transparent.color_batch_size')
        self.mask_batch_size = self.conf.get_int('optim_transparent.mask_batch_size')
        self.report_freq = self.conf.get_int('optim_transparent.report_freq')
        self.save_freq = self.conf.get_int('optim_transparent.save_freq')
        self.val_freq = self.conf.get_int('optim_transparent.val_freq')
        self.val_mesh_freq = self.conf.get_int('optim_transparent.val_mesh_freq')
        self.loss_type = self.conf.get_string('optim_transparent.loss_type',default='l1')
        
        params_to_train = []
        params_to_train += list(self.obj_network.parameters())
        
        
        self.ior_lr = self.conf.get_float('optim_transparent.ior_lr',default=1e-6)
        self.init_ior = self.conf.get_float('optim_transparent.init_ior',default=1.5)
        self.renderer.objIOR = torch.tensor(self.init_ior,requires_grad=True).to(self.device)
        self.init_ior = torch.tensor(self.init_ior).to(self.device)
        self.ior_optimizer = torch.optim.Adam([self.renderer.objIOR], lr=self.ior_lr)
        self.ior_weight = self.conf.get_float('optim_transparent.ior_weight',default=0.1)
        
        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)

        self.val_error = val_error
        if self.val_error:
            self.normal_path = os.path.join(self.dataset.data_dir,'normal')
            self.load_normal()
        
        self.base_exp_dir = os.path.join(self.base_exp_dir,self.exp_name)
        os.makedirs(self.base_exp_dir,exist_ok=True)
        self.file_backup()
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.iter_step = 0

    
    def load_normal(self):
        normal_lis = sorted(glob(self.normal_path+'/*.png'))
        normals_np = np.stack([(cv.imread(name))[:,:,::-1].astype(np.float32)/255*2-1 for name in normal_lis])
        self.normals = torch.from_numpy(normals_np.astype(np.float32)).cpu()  # [n_images, H, W, 1]
    
    def optim_transparent(self):

        self.update_learning_rate()

        image_perm = self.get_image_perm()
        self.validate_image()
        pred_point = self.validate_mesh()
        
        if self.val_error:
            gt_mesh_path = os.path.join(self.dataset.data_dir,'gt.ply')
            gt_mesh = trimesh.load(gt_mesh_path)
            gt_point = torch.tensor(np.array(gt_mesh.vertices),device='cuda')
            chamfer_dis = chamfer_distance_kaolin(pred_point, gt_point)
            self.writer.add_scalar('Static/chamfer_distance', chamfer_dis, self.iter_step)
        
        for iter_i in tqdm(range(self.iter_step,self.end_iter)):
            sigma = self.init_sigma / 2**(iter_i//self.coarse_level)
            self.scene_render.set_gaussian_sigma(sigma)
            
            # sample ray for color loss
            color_data, _ = self.dataset.gen_random_inside_rays_at(image_perm[self.iter_step % len(image_perm)], self.color_batch_size)
            rays_o, rays_d, true_rgb, gt_mask = color_data[:, :3], color_data[:, 3: 6], color_data[:, 6: 9], color_data[:, 9: 10]
            
            render_out = self.renderer.render_transparent(rays_o, rays_d)


            color = render_out['color']
            gradient_error = render_out['gradient_error']
            pred_mask = render_out['mask']
            
            # The rendering loss would gradually increase during optimization. 
            # This is because of the coarse-to-fine texture blur, and this is normal
            if self.loss_type=='l1':
                color_loss = F.l1_loss(color[pred_mask],true_rgb[pred_mask],reduction='mean')
            else:
                color_loss = F.mse_loss(color[pred_mask],true_rgb[pred_mask],reduction='mean')    
            eikonal_loss1 = gradient_error

            #sample ray for mask loss        
            mask_data = self.dataset.gen_random_margin(self.mask_batch_size)
            rays_o, rays_d, gt_mask = mask_data[:, :3], mask_data[:, 3: 6], mask_data[:, 6: 9]
            
            
            render_out = self.renderer.tracer.ray_tracing(rays_o, rays_d, tracing_method='volume')
            weight_sum = render_out['weights_sum']
            
            mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), gt_mask)
            eikonal_loss2 = render_out['gradient_error']
            if self.color_weight>0:
                num1 = (pred_mask.sum()+1e-5)
                num2 = eikonal_loss2.shape[0]
                eikonal_loss = eikonal_loss1.sum()/(num1+num2) + eikonal_loss2.sum()/(num1+num2)
            else:
                eikonal_loss = eikonal_loss2.mean()
            
            if pred_mask.sum()>1:
                assert (not torch.isnan(color_loss))
                assert (not torch.isnan(eikonal_loss))
        

            ior_loss = F.mse_loss(self.renderer.objIOR,self.init_ior)

            if self.color_weight>0:
                loss = color_loss * self.color_weight +\
                    eikonal_loss * self.igr_weight +\
                    mask_loss * self.mask_weight +\
                    ior_loss * self.ior_weight
                    
            else:
                loss = eikonal_loss * self.igr_weight +\
                    mask_loss * self.mask_weight +\
                    ior_loss * self.ior_weight
                    
            self.ior_optimizer.zero_grad()
            self.optimizer.zero_grad()
            
            if pred_mask.sum()>0:
                loss.backward(retain_graph=True)
                
                for param in self.obj_network.parameters():
                    assert (not torch.isnan(param.grad).any())
                self.ior_optimizer.step()
                self.optimizer.step()
                
            self.iter_step += 1
            
            start_time = time.time()
            self.writer.add_scalar('Loss/loss', loss, self.iter_step)
            self.writer.add_scalar('Loss/color_loss', color_loss, self.iter_step)
            self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
            self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
            self.writer.add_scalar('Loss/ior_loss', ior_loss, self.iter_step)
            
            
            
            self.writer.add_scalar('Static/sigma',sigma,self.iter_step)
            self.writer.add_scalar('Static/s_val', render_out['s_val'].mean(), self.iter_step)
            self.writer.add_scalar('Static/valid_ray', pred_mask.sum().item(), self.iter_step)
            self.writer.add_scalar('Static/lr', self.optimizer.state_dict()['param_groups'][0]['lr'], self.iter_step)
            self.writer.add_scalar('Static/ior_lr', self.ior_optimizer.state_dict()['param_groups'][0]['lr'], self.iter_step)
            self.writer.add_scalar('Static/ior', self.renderer.objIOR.item(), self.iter_step)
                    
            
            if self.iter_step % self.report_freq == 0:
                print('iter {:8>d}: loss={} color_loss={} mask_loss={} eikonal_loss={} \
                    lr={} mask_sum={}'.format(self.iter_step, loss.item(), color_loss.item(), mask_loss.item(), eikonal_loss.item(), self.optimizer.param_groups[0]['lr'], pred_mask.sum().item()))

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_freq == 0:
                self.validate_image()

            if self.iter_step % self.val_mesh_freq == 0:
                pred_point = self.validate_mesh()
                
                if self.val_error:
                    chamfer_dis = chamfer_distance_kaolin(pred_point, gt_point)
            
                    self.writer.add_scalar('Static/chamfer_distance', chamfer_dis, self.iter_step)
                
            self.update_learning_rate()
        
        if self.val_error:
            pred_point = self.validate_mesh(resolution=512)
            chamfer_dis = chamfer_distance_kaolin(pred_point, gt_point)
            self.writer.add_scalar('Static/chamfer_distance', chamfer_dis, self.iter_step)

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def update_learning_rate(self):
        
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
        
        for g in self.ior_optimizer.param_groups:
            g['lr'] = self.ior_lr * learning_factor

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py' and (not f_name.startswith('tmp')):
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))
        print('File Backup Done!')
        print('------------------------------------------------------')

    def find_latest_ckpt(self,path):
        model_list = sorted(glob(os.path.join(path,'*.pth')))
        return model_list[-1]

        
    def reload_trained_model(self):
        # Load checkpoint of stage1 for scene network
        texture_path = self.conf['renderer']['scene_renderer'].get_string('texture_path', default = '')
        if not texture_path:
            texture_path =  os.path.join(os.path.join(self.base_exp_dir,'stage1','export_texture','texture.png'))
        img = cv.imread(texture_path).astype(np.float32)/255.0
        texture = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)
        self.scene_render.set_texture_map(texture)
        
        # reload the initialized object model
        obj_model_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir,'init_shape','checkpoints'))
        logging.info('Load trained object model from: {}'.format(obj_model_path))
        ckpoint2 = torch.load(obj_model_path, map_location=self.device)
        self.tracer.sdf_network.load_state_dict(ckpoint2['obj_network'])
        self.tracer.deviation_network.load_state_dict(ckpoint2['obj_deviation_network'])
        

    def load_checkpoint(self):
        checkpoint_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir, 'checkpoints'))
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.obj_network.load_state_dict(checkpoint['obj_network'])
        self.obj_deviation_network.load_state_dict(checkpoint['obj_deviation_network:'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.iter_step = checkpoint['iter_step']
        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'obj_network': self.obj_network.state_dict(),
            'obj_deviation_network:': self.obj_deviation_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def cal_normal_error(self,pred,gt):
        error = np.clip((pred * gt).sum(axis=2),-1.0+1e-4,1.0-1e-4)
        degree_error = np.arccos(error) * 180 / np.pi
        return degree_error
        

    def validate_image(self, idx=-1, resolution_level=1):
        
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)
        
        if self.val_error:
            normal_gt = self.normals[idx].detach().cpu().numpy()
        
        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape

        rays_o = rays_o.reshape(-1, 3).split(self.color_batch_size//2)
        rays_d = rays_d.reshape(-1, 3).split(self.color_batch_size//2)

        out_rgb_fine = []
        out_normal_fine = []
        depth_list = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):

            render_out = self.renderer.render_with_transparent(rays_o_batch,
                                              rays_d_batch,
                                              )

            normals = render_out['normals'].detach().cpu().numpy()
            out_normal_fine.append(normals)

            color = render_out['color_fine'].detach().cpu().numpy()
            out_rgb_fine.append(color)
            
            
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 256).clip(0, 255)

        normal_pred = None
        normal_pred = np.concatenate(out_normal_fine, axis=0)
        rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
        normal_pred = np.matmul(rot[None, :, :], normal_pred[:, :, None])
        normal_pred[:,1] = -normal_pred[:,1]
        normal_pred[:,2] = -normal_pred[:,2]
        
        normal_pred = normal_pred.reshape(H,W,-3)
        
        if self.val_error:
            error = self.cal_normal_error(normal_gt,normal_pred)
            mask = normal_pred.sum(axis=2)==0
            error_map = colormap(error,30)
            error_map[mask,:] = 0
        
        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)
        

        for i in range(img_fine.shape[-1]):
            cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
            cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           (normal_pred[:,:,::-1]+1)/2*255)
            if self.val_error:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_gt_{}_{}.png'.format(self.iter_step, i, idx)),
                        (normal_gt[:,:,::-1]+1)/2*255)
                
                cv.imwrite(os.path.join(self.base_exp_dir,
                                            'normals',
                                            '{:0>8d}_error_{}_{}.png'.format(self.iter_step, i, idx)),
                        (error_map*255))
            
    def validate_mesh(self, resolution=256, threshold=0.0):
        bounds = enlarge_bounding_box(self.bounds)
        bound_min = torch.from_numpy(bounds[0,:])
        bound_max = torch.from_numpy(bounds[1,:])

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)
        mesh = trimesh.Trimesh(vertices, triangles)
        mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))
        # return mesh
        return torch.tensor(np.array(mesh.vertices),device='cuda')
       

if __name__ == '__main__':
    print('Hello Ark')
    print('\n')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/optim_trans.conf') 
    parser.add_argument('--exp_name', type=str, default='optim_trans')    
    parser.add_argument('--extra_name', type=str, default='')    
    parser.add_argument('--gpu', type=int, default=0)                   
    parser.add_argument('--case', type=str, default='pig')                  
    parser.add_argument('--val_error', action='store_true')
    
    args = parser.parse_args()
    if args.extra_name:
        args.exp_name += '_{}'.format(args.extra_name)

    print("Deal case {} \n".format(args.case))
    torch.cuda.set_device(args.gpu)
    exper = Exper(args.conf, args.case, args.exp_name, args.val_error)
    
    exper.optim_transparent()
    


