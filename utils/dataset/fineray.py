from torch.utils.data import Dataset
import numpy as np
import os

class FineRayDataset(Dataset):
    """ dataset of biasly sampled nerf-synthetic rays according to coarse occupancy map"""
    def __init__(self, root_dir):
        self.root_dir = os.path.join(root_dir,'fine_rays.npz')
        fine_rays = np.load(self.root_dir)
        
        self.rays = fine_rays['rays']
        self.rgbs = fine_rays['rgbs']

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, idx):
        ray = self.rays[idx]
        rgb = self.rgbs[idx]
        return {'rays': ray, 'rgbs': rgb}