import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T




class TanksDataset(Dataset):
    """ dataset for TnT and BlendedMVS"""
    def __init__(self, root_dir, split='train',img_wh=(1920,1080),
                 sub_samples=2048,sample_nums=10,mask_path=None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.define_transforms()
        self.sub_samples = sub_samples
        self.sample_nums = sample_nums
        self.mask_path = mask_path
        self.read_meta()
        
    def define_transforms(self):
        self.transform = T.ToTensor()
        
    def read_meta(self):
        intrinsics = load_intrinsics(os.path.join(self.root_dir,'intrinsics.txt'))
        fx,fy,cx,cy = parse_intrinsics(intrinsics)
        w,h = self.img_wh

        dataset_name = self.root_dir.split('/')[-2]
        if dataset_name == 'BlendedMVS':
            self.fx = fx * w/768.
            self.fy = fy * h/576.
            self.cx = cx * w/768.
            self.cy = cy * h/576.
        elif dataset_name == 'TanksAndTemple':
            self.fx = fx * w/1920.
            self.fy = fy * h/1080.
            self.cx = cx * w/1920.
            self.cy = cy * h/1080.

        #read bounding box xyzmin,xyzmax
        with open(os.path.join(self.root_dir,'bbox.txt'),'r') as file:
            self.bounds = [float(x) for x in file.readlines()[0].strip().split()[:6]]

            
        # center and normalize the boundary
        xmin,ymin,zmin,xmax,ymax,zmax = self.bounds
        self.mid_point = torch.FloatTensor([(xmin+xmax)/2,(ymin+ymax)/2,(zmin+zmax)/2])[None]
        self.scale = max(xmax-xmin,ymax-ymin,ymax-zmin)*0.5

            
            
        self.directions = get_ray_directions(h,w,self.fx,self.fy,self.cx,self.cy)

        prefix = '0' if self.split == 'train' or self.split == 'extra'  else '1'

        img_paths = os.path.join(self.root_dir,'rgb')
        img_files = [f for f in os.listdir(img_paths) if f[0] == prefix]
        pose_files = [f.replace('.png','.txt') for f in img_files]
        img_paths = [os.path.join(img_paths,f) for f in img_files]
        
        pose_paths = os.path.join(self.root_dir,'pose')
        pose_paths = [os.path.join(pose_paths,f) for f in pose_files]

        poses = [parse_extrinsics(load_matrix(f))[:3] for f in pose_paths]
        self.image_paths = img_paths
        self.pose_paths = pose_paths
        self.poses = poses

    def __len__(self):
        if self.split == 'train':
            return len(self.image_paths)*self.sample_nums
        if self.split == 'val':
            return 8 # only validate 8 images (to support <=8 gpus)
        return len(self.image_paths)

    def __getitem__(self, idx):
        
        if self.split == 'train':
            idx = idx % len(self.image_paths)
        
        pose = self.poses[idx]
        image_path = self.image_paths[idx]
        c2w = torch.FloatTensor(pose)

        img = Image.open(image_path)
        img = img.resize(self.img_wh, Image.LANCZOS)
        img = self.transform(img) # (4, H, W)
        if img.shape[0] == 4:
            img = img[:3]*img[3:]+(1-img[3:])
        img = img.view(3, -1).permute(1, 0) # (H*W, 4) RGB

        rays_o, rays_d = get_rays(self.directions, c2w)

        rays_o = (rays_o-self.mid_point)/self.scale

        rays = torch.cat([rays_o, rays_d],1)
        
        if self.split == 'train': # use data in the buffers
            if self.mask_path is not None:
                mask_img = Image.open(os.path.join(self.mask_path,'mask_{}.png'.format(idx)))
                mask_img = self.transform(mask_img)[0].reshape(-1)>0
                rays = rays[mask_img]
                img = img[mask_img]
            idxs = torch.randperm(len(rays))[:self.sub_samples]
            sample = {'rays': rays[idxs],
                      'rgbs': img[idxs]}

        else: # create data for each image separately

            sample = {'rays': rays,
                      'rgbs': img,
                      'c2w': c2w}

        return sample
    
def get_ray_directions(H, W, fx,fy,cx,cy):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        H, W, focal: image height, width and focal length

    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    #grid = create_meshgrid(H, W, normalized_coordinates=False)[0]
    #i, j = grid.unbind(-1)

    O = 0.5
    x_coords = torch.linspace(O, W - 1 + O, W)
    y_coords = torch.linspace(O, H - 1 + O, H)
    j, i = torch.meshgrid([y_coords, x_coords])
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    directions = \
        torch.stack([(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1) # (H, W, 3)

    return directions

def get_rays(directions, c2w):
    """
    Get ray origin and normalized directions in world coordinate for all pixels in one image.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems

    Inputs:
        directions: (H, W, 3) precomputed ray directions in camera coordinate
        c2w: (3, 4) transformation matrix from camera coordinate to world coordinate

    Outputs:
        rays_o: (H*W, 3), the origin of the rays in world coordinate
        rays_d: (H*W, 3), the normalized direction of the rays in world coordinate
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:, :3].T # (H, W, 3)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    # The origin of all rays is the camera origin in world coordinate
    rays_o = c2w[:,3].expand(rays_d.shape) # (H, W, 3)

    rays_d = rays_d.view(-1, 3)
    rays_o = rays_o.view(-1, 3)

    return rays_o, rays_d


def parse_extrinsics(extrinsics, world2camera=False):
    """ this function is only for numpy for now"""
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
        extrinsics = extrinsics.reshape(4, 4)
    if world2camera:
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
    extrinsics[:3, 1:3] = -extrinsics[:3, 1:3]
    return extrinsics

def load_matrix(path):
    return np.array([[float(w) for w in line.strip().split()] for line in open(path)]).astype(np.float32)   

def parse_intrinsics(intrinsics):
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    return fx, fy, cx, cy

def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        return intrinsics
    except ValueError:
        pass
    
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    