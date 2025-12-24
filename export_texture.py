import cv2 
import os
import logging
import argparse
import numpy as np
import random
from glob import glob
import torch
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer


SEED = 0
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)


def get_texture(renderer, resolution=512):
    """
    With the recovered NeuS (and NeRF++) in stage1, we render the plane from top view to get its appearance as texture
    """
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

    for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
        render_output = renderer.render(rays_o_batch,
                                            rays_d_batch,
                                            sample_dir=None
                                            )
        color_list.append(render_output['color'].detach().cpu().numpy())

    color = np.concatenate(color_list,axis=0).reshape(resolution,resolution,3).astype(np.float32)
        
    return color

class Exper:
    """
        Load the trained model in stage1 and render the plane texture
    """
    def __init__(self, conf_path, case='CASE_NAME', exp_name='export_texture'):
        self.exp_name = exp_name
        self.device = torch.device('cuda')
        
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

        self.use_white_bkgd = False
        
        self.model_list = []

        # Networks
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        

        self.renderer = NeuSRenderer(self.nerf_outside,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    **self.conf['model.neus_renderer'])

        # reload the trained NeuS in stage1
        self.reload_trained_model()

        
    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def find_latest_ckpt(self,path):
        model_list = sorted(glob(os.path.join(path,'*.pth')))
        return model_list[-1]

    def reload_trained_model(self):
        # Load checkpoint of stage1 for scene network
        stage1_model_path = self.find_latest_ckpt(os.path.join(self.base_exp_dir,'checkpoints'))
        logging.info('Load trained stage1 model from: {}'.format(stage1_model_path))
        checkpoint = torch.load(stage1_model_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])

    def render_desktop(self):
        basedir = os.path.join(self.base_exp_dir,self.exp_name)
        os.makedirs(basedir,exist_ok=True)
        resolution = 512
        texture = get_texture(self.renderer, resolution=resolution)
        cv2.imwrite(os.path.join(basedir,'texture.png'),texture*255)


if __name__ == '__main__':
    print('Hello Ark')

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/stage1.conf')    
    parser.add_argument('--gpu', type=int, default=0)                      
    parser.add_argument('--case', type=str, default='pig')                 

    args = parser.parse_args()

    print("Deal case {}".format(args.case))
    torch.cuda.set_device(args.gpu)
    exper = Exper(args.conf, args.case)
    
    exper.render_desktop()
    
