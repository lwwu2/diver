from torch.utils.cpp_extension import load
from torch.autograd import Function
from pathlib import Path

_ext_src_root = Path(__file__).parent / 'src'
_ext_include = Path(__file__).parent / 'include'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

_ext = load(name='ray_voxel_ext', 
            sources=_ext_src_files, 
            extra_include_paths=[str(_ext_include)])


class RayVoxelIntersect(Function):
    @staticmethod
    def forward(ctx, o, v, xyzmin, xyzmax, voxel_num, voxel_size):
        """ training ray marching code
        Args:
            o: Bx3 ray origin
            v: Bx3 ray direction
            xyzmin: min volume bound
            xyzmax: max volume bound
            voxel_num: voxel grid size
            voxel_size: size of each voxel
        Return:
            - BxKx3 intersected points
            - BxK hit indicator
            - BxK distance from the ray origin
        """
        intersection, intersect_num, tns = _ext.ray_voxel_intersect(
            o, v,
            xyzmin,xyzmax,
            voxel_num, voxel_size
        )
        
        # free space where every pixel does not have hits
        num_max = intersect_num.max()
        intersection = intersection[:, :num_max]
        tns = tns[:, :num_max]
        return intersection,(tns>=0), tns

    @staticmethod
    def backward(ctx, a, b, c, d, e, f):
        return None, None, None, None, None, None
    

class MaskedIntersect(Function):
    @staticmethod
    def forward(ctx, o, v, xyzmin, xyzmax, voxel_num, voxel_size, mask, mask_scale):
        """ training ray marching with occupancy mask
        Args:
            o: Bx3 ray origin
            v: Bx3 ray direction
            xyzmin: min volume bound
            xyzmax: max volume bound
            voxel_num: voxel grid size
            voxel_size: size of each voxel
            mask: (Nxmask_scale)**3  occupancy mask
            mask_scale: relative scale of the occupancy mask in respect to the voxel grid
        Return:
            - BxKx6 intersected entry + exit point
            - BxK hit indicator
            - BxK distance from the ray origin
        """
        intersection, intersect_num, tns = _ext.masked_intersect(
            o, v, mask,
            xyzmin, xyzmax,
            voxel_num, voxel_size, mask_scale
        )
        
        # free space where every pixel does not have hits
        num_max = intersect_num.max()
        intersection = intersection[:, :num_max]
        tns = tns[:, :num_max]

        return intersection,(tns>=0), tns

    @staticmethod
    def backward(ctx, a, b, c, d, e, f, g, h):
        return None, None, None, None, None, None, None, None
    

ray_voxel_intersect = RayVoxelIntersect.apply
masked_intersect = MaskedIntersect.apply