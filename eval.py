import torch
import torch.nn.functional as NF
from torchvision.utils import save_image
import math
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from configs.config import default_options
from model.diver import DIVeR

from utils.dataset import BlenderDataset, TanksDataset


def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser

""" evaluate the test sequence for offline rendering, store the rendered image in output folder"""
if __name__ == '__main__':
    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()
    
    # add PROGRAM level args
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--device', type=int, required=False,default=None)
    parser.add_argument('--batch', type=int, default=20480)
    
    parser.set_defaults(resume=False)
    args = parser.parse_args()

    
    device = torch.device(args.device)
    
    checkpoint_path = Path(args.checkpoint_path)
    # load model
    if args.checkpoint_path[-4:] == '.pth': # model weight file
        hparams.mask_scale = 1 # assume the occupancy mask is at fine scale
        weight = torch.load(checkpoint_path,map_location='cpu')
        weight['voxels'] = weight['voxels'].to_dense()
        model = DIVeR(hparams)
        with torch.no_grad():
            model.init_voxels(False)

        model.load_state_dict(weight,strict=False)
    else: # model checkpoint point file
        state_dict = torch.load(checkpoint_path, map_location='cpu')['state_dict']
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
    elif dataset_name == 'tanks':
        dataset_fn = TanksDataset

    dataset = dataset_fn(dataset_path,img_wh=hparams.im_shape[::-1], split='test')
    batch_size = args.batch
    
    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    psnrs = []
    idx = 0
    for batch in tqdm(dataset):
        rays,rgb_h = batch['rays'].to(device), batch['rgbs'].to(device)
    
        rgbs = []
        depths = []

        for b_id in range(math.ceil(rays.shape[0]*1.0/batch_size)):
            b_min = b_id*batch_size
            b_max = min(rays.shape[0],(b_id+1)*batch_size)
            x,d = rays[b_min:b_max,:3],rays[b_min:b_max,3:6]

            color, sigma, mask, ts = model(x,d)
            if color is None:
                rgb = torch.ones(mask.shape[0],3,device=device)
                depth = torch.zeros(mask.shape[0],device=device)
            else:
                rgb, weight = model.render(color, sigma, mask)   
                depth = (ts*mask*weight).sum(1)
            
            rgbs.append(rgb)
            depths.append(depth)

        rgbs = torch.cat(rgbs,0)
        depths = torch.cat(depths,0)
        depths /= (depths.max()+1e-3)
        psnr = -10.0 * math.log10(NF.mse_loss(rgbs,rgb_h).clamp_min(1e-5))
        save_image(rgbs.T.reshape(3,*hparams.im_shape), output_path/'{:03d}.png'.format(idx))
        save_image(depths.reshape(*hparams.im_shape), output_path/'depth{:03d}.png'.format(idx))
        idx += 1
        psnrs.append(psnr)

    psnrs = torch.tensor(psnrs)
    # report average PSNR
    print('PSNR: {}'.format(psnrs.mean().item()))