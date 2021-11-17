import torch
import torch.nn.functional as NF
from torchvision.utils import save_image
import math
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

import time

from configs.config import default_options

from utils.dataset import BlenderDataset
from utils.ray_march import aabb_intersect,ray_march


def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
    
""" evaluate the test sequence for real-time rendering """
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    
    
    # add PROGRAM level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--decoder', type=int ,default=32) # deoder architecture
    
    
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    if args.decoder == 32: # select decoder function
        from utils.mlp_evaluation import mlp_eval, upload_weight
    else:
        from utils.mlp_evaluation64 import mlp_eval, upload_weight
    
    device = torch.device(args.device)
    
    checkpoint_path = Path(args.checkpoint_path)
    # load serialized model
    weight = torch.load(checkpoint_path, map_location='cpu')
    params = weight['p'].contiguous() # padd to fit in a warp

    inds = weight['i'].long()
    m_mask = weight['m']
    ii = inds // ((hparams.voxel_num+1)*(hparams.voxel_num+1))
    inds = inds % ((hparams.voxel_num+1)*(hparams.voxel_num+1))
    jj = inds // (hparams.voxel_num+1)
    kk = inds % (hparams.voxel_num+1)
    
    idxs = torch.stack([ii,jj,kk],0)
    features = weight['f']
    voxel_masks = torch.zeros(hparams.voxel_num,\
                hparams.voxel_num,hparams.voxel_num,dtype=bool)
    voxel_masks[ii[m_mask],jj[m_mask],kk[m_mask]] = True
    voxel_masks = voxel_masks.to(device).contiguous()


    voxel_map = torch.zeros([hparams.voxel_num+1]*3,dtype=torch.int)
    voxel_map[idxs[0],idxs[1],idxs[2]] = torch.arange(len(idxs[0])).int()
    voxels = features.to(device).contiguous()
    voxel_map = voxel_map.contiguous()
    
    
    level = 3
    scales = [4**i for i in range(1,1+level)]
    octrees = [NF.max_pool3d(voxel_masks[None,None].float(),s,stride=s)[0,0].bool().reshape(-1) for s in scales[::-1]]
    octrees = torch.cat(octrees,0).contiguous()
    
    params = params.contiguous()
    upload_weight(device.index,params,voxel_map)
    
    voxel_size = hparams.grid_size/hparams.voxel_num
    xyzmin = -hparams.voxel_num*voxel_size*0.5
    voxel_num = hparams.voxel_num
    
    # allocate buffer
    coords = torch.zeros(8,*hparams.im_shape,6).to(device).contiguous()
    rgba = torch.zeros(*hparams.im_shape,4).to(device).contiguous()
    finish = torch.zeros(*hparams.im_shape,dtype=bool).to(device).contiguous()
    
    # load dataset
    dataset_name,dataset_path = hparams.dataset
    if dataset_name == 'blender':
        dataset_fn = BlenderDataset

    dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='test')
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    psnrs = []
    times = []
    idx = 0
    for batch in tqdm(dataset):
        rays,rgb_h = batch['rays'], batch['rgbs']
        
        v = rays[:,3:6].reshape(*hparams.im_shape,3).to(device).contiguous()
        center = (rays[0,:3]-xyzmin)/voxel_size
        center = center.to(device).contiguous()
        
        # recorder evaluation time for each frame
        start_time = time.time()
        aabb_intersect(coords,v,center,finish,rgba,octrees,voxel_num)
        while not finish.all():
            ray_march(coords,v,voxel_masks,finish,voxel_num)
            mlp_eval(rgba, coords, voxels, v, finish)
        total_time = time.time()-start_time
        times.append(total_time)
        
        rgbs = rgba.cpu()[:,:,:3]+rgba.cpu()[:,:,3:]
        rgbs = rgbs.reshape(-1,3)
        
        psnr = -10.0 * math.log10(NF.mse_loss(rgbs,rgb_h).clamp_min(1e-5))
        save_image(rgbs.T.reshape(3,*hparams.im_shape), output_path/'{:03d}.png'.format(idx))
        idx += 1
        psnrs.append(psnr)
        

    psnrs = torch.tensor(psnrs)
    times = torch.tensor(times)
    # report average PSNR and FPS
    print('PSNR: {}'.format(psnrs.mean().item()))
    print('Time: {}, FPS: {}'.format(times.mean().item(),1.0/times.mean().item()))