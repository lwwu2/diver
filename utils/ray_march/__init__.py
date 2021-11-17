from torch.utils.cpp_extension import load
from pathlib import Path

_ext_src_root = Path(__file__).parent / 'src'
_ext_include = Path(__file__).parent / 'include'
exts = ['.cpp', '.cu']
_ext_src_files = [
    str(f) for f in _ext_src_root.iterdir() 
    if any([f.name.endswith(ext) for ext in exts])
]

_ext = load(name='raymarching_ext', 
            sources=_ext_src_files, 
            extra_include_paths=[str(_ext_include)])

ray_march = _ext.ray_march
aabb_intersect = _ext.aabb_intersect