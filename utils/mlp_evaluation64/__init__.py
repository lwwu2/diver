from torch.utils.cpp_extension import load
import torch
from torch.autograd import Function
from pathlib import Path

_ext_src_root = Path(__file__).parent / 'src'
_ext_include = Path(__file__).parent / 'include'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

_ext = load(name='mlp_eval_ext', 
            sources=_ext_src_files, 
            extra_include_paths=[str(_ext_include)])

class MLPEval(Function):
    @staticmethod
    def forward(ctx, rgba, coord, voxels, v, mask):
        _ext.mlp_eval(
            rgba,
            coord,voxels,v,mask)
        return torch.Tensor([0])

    @staticmethod
    def backward(ctx, a, b, c, d, e):
        return None, None, None, None, None
    
class UploadWeight(Function):
    @staticmethod
    def forward(ctx, device_id, params, voxel_map):
        _ext.upload_weight(
            device_id,
            params, voxel_map)
        
        return torch.Tensor([0])

    @staticmethod
    def backward(ctx, a, b, c):
        return None, None, None
    
mlp_eval = MLPEval.apply
upload_weight = UploadWeight.apply