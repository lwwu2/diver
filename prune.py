import torch
import torch_scatter
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import numpy as np
import math

from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import sys
sys.path.append('..')
from configs.config import default_options
from model.diver import DIVeR

from utils.dataset import BlenderDataset,TanksDataset
from utils.ray_voxel_intersection import ray_voxel_intersect, masked_intersect

def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    
    # add PROGRAM level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--thresh', type=float, default=1e-2) # voxel culling threshold
    parser.add_argument('--pad' , type=int, default=0) # whether to padd the boundary
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--batch', type=int, default=4000)
    parser.add_argument('--bias_sampling', type=int, default=0) # whether bias sampling the fine_rays

    parser.set_defaults(resume=False)
    args = parser.parse_args()

    
    device = torch.device(args.device)
    checkpoint_path = Path(args.checkpoint_path)
    
    # load model
    state_dict = torch.load(checkpoint_path/'last.ckpt', map_location='cpu')['state_dict']
    weight = {}
    for k,v in state_dict.items():
        if 'model.' in k:
            weight[k.replace('model.', '')] = v

    model = DIVeR(hparams)
    if hparams.implicit:
        with torch.no_grad():
            model.init_voxels(False)
    model.load_state_dict(weight, strict=True)

    model.to(device)
    for p in model.parameters():
        p.requires_grad = False

    
    # load dataset
    dataset_name,dataset_path = hparams.dataset
    batch_size = args.batch
    if dataset_name == 'blender':
        dataset_fn = BlenderDataset
        dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    elif dataset_name == 'tanks':
        dataset_fn = TanksDataset
        dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='extra')
        dataloader = dataset
    


    alpha_map_path = checkpoint_path / 'alpha_map.pt'

    if True: #not alpha_map_path.exists():
        # extracting alpha map
        print('extracting alpha map')
        alpha_map = torch.zeros((model.voxel_num)**3,device=device)

        for batch in tqdm(dataloader):
            rays = batch['rays'].to(device)
            for b_id in range(math.ceil(len(rays)*1.0/batch_size)):
                b_min = b_id*batch_size
                b_max = min((b_id+1)*batch_size,len(rays))
                xs, ds = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]

                # perform ray-voxel intersection
                if hasattr(model, 'voxel_mask'):
                    coord,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                                   model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                                   model.voxel_mask.contiguous(), model.mask_scale)
                    if not mask.any():
                        continue
                    coord=coord[mask]
                    coord_in = coord[:,:3]
                    coord_out = coord[:,3:]
                else:
                    coord, mask, _ = ray_voxel_intersect(xs.contiguous(), ds.contiguous(), model.xyzmin, model.xyzmax, int(model.voxel_num),model.voxel_size)
                    if not mask.any():
                        continue
                    mask = mask[:,:-1]&mask[:,1:]
                    coord_in = coord[:,:-1][mask]
                    coord_out = coord[:,1:][mask]

                # get accumulated alphas
                color,sigma = model.decode(coord_in, coord_out, ds,mask)
                _, weight = model.render(color, sigma, mask)
                weight = weight[mask]

                # accurate voxel corner calculation
                coord = torch.min((coord_in+1e-4).long(),(coord_out+1e-4).long())

                # check if out of boundary
                bound_mask = ((coord>=args.pad) & (coord<=model.voxel_num-1-args.pad)).all(-1)
                coord = coord[bound_mask]
                weight = weight[bound_mask]

                # flattened occupancy mask index
                coord = coord[:,0] + coord[:,1]*model.voxel_num + coord[:,2]*(model.voxel_num)**2

                # scatter-max to the occupancy map
                alpha_map = torch_scatter.scatter(
                        weight, coord, dim=0, out=alpha_map,reduce='max')

        alpha_map = alpha_map.reshape(model.voxel_num,model.voxel_num, model.voxel_num)
        torch.save(alpha_map.cpu(), alpha_map_path) # save in the model weight folder
    else:
        print('find existing alpha map')
        print('make sure it is the correctly baked one!')
        alpha_map = torch.load(alpha_map_path, map_location='cpu')

        
    voxel_mask = alpha_map.to(device) > args.thresh
    print('{} sapce preserved'.format(voxel_mask.float().mean().item()))

    if args.bias_sampling==0: # exit if not bias sampling
        exit(0)
    
          
    print('extract fine rays')
    fine_rays = []
    fine_rgbs = []
    
    batch_size *= 10
    # bias sampling the fine rays 
    if dataset_name == 'blender': # for nerf-synthetic, we directly store the sampled rays as .npz file in the weight folder
        dataset_fn = BlenderDataset
        fine_size = [800,800]
        dataset = dataset_fn(dataset_path,img_wh=fine_size[::-1], split='train')
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
        for batch in tqdm(dataloader):
            rays = batch['rays'].to(device)
            rgbs = batch['rgbs'].to(device)

            xs,ds = rays[:,:3],rays[:,3:6]
            _,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                        model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                        voxel_mask.contiguous(), model.mask_scale)
            valid_rays = mask.any(1)

            if valid_rays.any():
                fine_rays.append(rays[valid_rays].cpu())
                fine_rgbs.append(rgbs[valid_rays].cpu())

        fine_rays = torch.cat(fine_rays,0)
        fine_rgbs = torch.cat(fine_rgbs,0)

        print('{} rays perserved'.format(len(fine_rays)*1.0/len(dataset)))

        fine_path = checkpoint_path / 'fine_rays.npz'
        np.savez(fine_path, rays=fine_rays, rgbs=fine_rgbs)
    
    elif dataset_name == 'tanks': # for tnt, blendedmvs, we store the pixel(ray) mask
        if 'BlendedMVS' in dataset_path:
            fine_size = [576,768]
        elif 'TanksAndTemple' in dataset_path:
            fine_size = [1080,1920]
        
        dataset_fn = TanksDataset
        dataset = dataset_fn(dataset_path,img_wh=fine_size[::-1], split='extra')
        dataloader = dataset
        
        mask_path = checkpoint_path / 'masks'
        mask_path.mkdir(parents=True, exist_ok=True)
        
        idx = 0
        for batch in tqdm(dataloader):
            masks = []
            rays = batch['rays'].to(device)
            rgbs = batch['rgbs'].to(device)
            
            for b_id in range(math.ceil(len(rays)*1.0/batch_size)):
                b_min = b_id*batch_size
                b_max = min((b_id+1)*batch_size,len(rays))
                xs, ds = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]
                
                _,mask,_ = masked_intersect(xs.contiguous(), ds.contiguous(),\
                                        model.xyzmin, model.xyzmax, int(model.voxel_num), model.voxel_size,\
                                        voxel_mask.contiguous(), model.mask_scale)
                mask = mask.any(1)
                masks.append(mask.cpu())
            
            masks = torch.cat(masks,-1).reshape(fine_size).float()
            save_image(masks, mask_path/'mask_{}.png'.format(idx)) # save in the mask folder inside model weight folder
            idx += 1