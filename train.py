import torch
import torch.nn.functional as NF
import torch.optim as optim
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import math


import os
from pathlib import Path
from argparse import Namespace, ArgumentParser



from configs.config import default_options
from model.diver import DIVeR

from utils.dataset import BlenderDataset, TanksDataset
from utils.dataset.fineray import FineRayDataset



class ModelTrainer(pl.LightningModule):
    """ diver model training code """
    def __init__(self, hparams: Namespace, *args, **kwargs):
        super(ModelTrainer, self).__init__()
        self.save_hyperparameters(hparams)
        self.im_shape = hparams.im_shape
        self.model = DIVeR(self.hparams)

    def __repr__(self):
        return repr(self.hparams)

    def configure_optimizers(self):
        if(self.hparams.optimizer == 'SGD'):
            opt = optim.SGD
        if(self.hparams.optimizer == 'Adam'):
            opt = optim.Adam

        optimizer = opt(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)

        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=self.hparams.milestones,gamma=self.hparams.scheduler_rate)
        return [optimizer], [scheduler]
    
    def train_dataloader(self,):
        dataset_name,dataset_path = self.hparams.dataset
        if self.hparams.fine > 0: # fine training
            if dataset_name == 'blender':
                dataset = FineRayDataset(self.hparams.coarse_path)
            elif dataset_name == 'tanks':
                new_batch_size=max(1,self.hparams.batch_size//2048)
                dataset = TanksDataset(dataset_path,img_wh=(self.im_shape[1],self.im_shape[0]),
                                       sample_nums=new_batch_size*10, mask_path=os.path.join(self.hparams.coarse_path,'masks'))
                return DataLoader(dataset, shuffle=True,num_workers=self.hparams.num_workers,batch_size=new_batch_size)
        else:
            if dataset_name == 'blender':
                dataset = BlenderDataset(dataset_path,img_wh=self.im_shape)
            elif dataset_name == 'tanks':
                new_batch_size=max(1,self.hparams.batch_size//2048)
                dataset = TanksDataset(dataset_path,img_wh=(self.im_shape[1],self.im_shape[0]),sample_nums=new_batch_size*10)
                return DataLoader(dataset, shuffle=True,num_workers=self.hparams.num_workers,batch_size=new_batch_size)
        return DataLoader(dataset, shuffle=True,num_workers=self.hparams.num_workers,batch_size=self.hparams.batch_size)
    
    def val_dataloader(self,):
        dataset_name,dataset_path = self.hparams.dataset
        if dataset_name == 'blender':
            dataset = BlenderDataset(
                dataset_path,
                split='val',
                img_wh=self.im_shape
            )
        elif dataset_name == 'tanks':
                dataset = TanksDataset(dataset_path,split='val',img_wh=(self.im_shape[1],self.im_shape[0]))
        return DataLoader(dataset, shuffle=False)

    def forward(self, points, view):
        return

    def sigma_loss(self, sigma):
        return torch.log(1+sigma**2/0.5).mean()

    
    def training_step(self, batch, batch_idx):
        """ one training step
        Args:
            batch:
                - rays: Bx8 tensor of (ray origin, direction, near, far)
                - rgbs: Bx3 tensor of ground truth color
        Return:
            training loss
        """
        rays,rgb_h = batch['rays'], batch['rgbs']
        rays = rays.reshape(-1,rays.shape[-1])
        rgb_h = rgb_h.reshape(-1,3)
        x,d = rays[:,:3],rays[:,3:6]
        
        color, sigma, mask, _ = self.model(x, d)
        
        if color is None: # all the sampled rays missed 
            return None
        
        rgb, _ = self.model.render(color, sigma, mask)
        
        loss_c = NF.mse_loss(rgb, rgb_h) # reconstruction loss
        loss_reg = self.hparams.l_s*self.sigma_loss(sigma) # sparsity regularization loss

        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))
        loss = loss_c + loss_reg
        
        self.log('train/loss', loss)
        self.log('train/psnr', psnr)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """ one validation step, report rendered image and distance
        Args:
            batch:
                - rays: Bx8 tensor of (ray origin, direction, near, far)
                - rgbs: Bx3 tensor of ground truth color
        """

        rays,rgb_h = batch['rays'], batch['rgbs']
        rays = rays.reshape(-1,rays.shape[-1])
        rgb_h = rgb_h.reshape(-1,3)
        
        batch_size = self.hparams.batch_size*2 # batch size used for rendering

        rgbs,depths = [],[]
        for b_id in range(math.ceil(rays.shape[0]*1.0/batch_size)):
            x,d = rays[b_id*batch_size:(b_id+1)*batch_size,:3],rays[b_id*batch_size:(b_id+1)*batch_size,3:6]
            color, sigma, mask, ts = self.model(x,d)
            
            if color is None:
                rgb = torch.ones(mask.shape[0],3,device=mask.device)
                depth = torch.zeros(mask.shape[0],device=mask.device)
            else:
                rgb, weight = self.model.render(color, sigma, mask)
                rgb = rgb.clamp(0,1)
                depth = (ts*mask*weight).sum(1)
                
            rgbs.append(rgb)
            depths.append(depth)
            
        rgbs = torch.cat(rgbs,0)
        depths = torch.cat(depths,0)
        
        loss_c = NF.mse_loss(rgbs, rgb_h)
        loss = loss_c
        psnr = -10.0 * math.log10(loss_c.clamp_min(1e-5))

        
        # logging
        self.log('val/loss', loss)
        self.log('val/psnr', psnr)
        self.logger.experiment.add_image('val/gt_image', rgb_h.reshape(*self.im_shape,3).permute(2, 0, 1), batch_idx)
        self.logger.experiment.add_image('val/inf_image', rgbs.reshape(*self.im_shape,3).permute(2, 0, 1), batch_idx)
        self.logger.experiment.add_image('val/inf_dis', depths.reshape(*self.im_shape,1).permute(2,0,1).expand(3,*self.im_shape), batch_idx)
        
        return

            
def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        for name, args in default_options.items():
            if(args['type'] == bool):
                parser.add_argument('--{}'.format(name), type=eval, choices=[True, False], default=str(args.get('default')))
            else:
                parser.add_argument('--{}'.format(name), **args)
        return parser
        
if __name__ == '__main__':

    torch.manual_seed(9)
    torch.cuda.manual_seed(9)

    parser = ArgumentParser()
    parser = add_model_specific_args(parser)
    hparams, _ = parser.parse_known_args()

    # add PROGRAM level args
    parser.add_argument('--experiment_name', type=str, required=True)
    parser.add_argument('--max_epochs', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--ft', type=str, default=None)
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
    parser.add_argument('--resume', dest='resume', action='store_true')
    parser.add_argument('--device', type=int, required=False,default=None)

    parser.set_defaults(resume=False)
    args = parser.parse_args()
    args.gpus = [args.device]
    experiment_name = args.experiment_name

    # setup checkpoint loading
    checkpoint_path = Path(args.checkpoint_path) / experiment_name
    log_path = Path(args.log_path)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(checkpoint_path, monitor='val/loss', save_top_k=1, save_last=True)
    logger = TensorBoardLogger(log_path, name=experiment_name)

    last_ckpt = checkpoint_path / 'last.ckpt' if args.resume else None
    if (last_ckpt is None) or (not (last_ckpt.exists())):
        last_ckpt = None
    else:
        last_ckpt = str(last_ckpt)
    
    # setup model trainer
    model = ModelTrainer(hparams)
    
    if args.coarse_path is not None and last_ckpt is None: # load occupancy mask from coarse training
        coarse_voxel = Path(args.coarse_path) / 'alpha_map.pt'
        alpha_map = torch.load(coarse_voxel, map_location='cpu')
        voxel_mask = alpha_map > hparams.thresh_a
        model.model.voxel_mask.data = voxel_mask
    
    if args.ft is not None: # intiialize explicit grid from implicit MLP
        ft_state = torch.load(args.ft, map_location='cpu')['state_dict']
        ft_weight = {}
        for k,v in ft_state.items():
            if 'model.' in k:
                ft_weight[k.replace('model.','')] = v
        model.model.load_state_dict(ft_weight)

        with torch.no_grad(): # slow, move to gpu?
            model.model.init_voxels()
    
    trainer = Trainer.from_argparse_args(
        args,
        resume_from_checkpoint=last_ckpt,
        logger=logger,
        checkpoint_callback=checkpoint_callback,
        flush_logs_every_n_steps=1,
        log_every_n_steps=1,
        max_epochs=args.max_epochs
    )

    trainer.fit(model)