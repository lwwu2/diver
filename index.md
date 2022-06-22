&nbsp;
## Abstract
![pipeline](https://user-images.githubusercontent.com/93700419/142577732-791a87a4-8b1f-40e9-97f4-c8e7886c95fa.png)

DIVeR builds on the key ideas of NeRF and its variants -- density models and volume rendering -- to learn 3D object models that can be rendered realistically from small numbers of images.  In contrast to all previous NeRF methods, DIVeR uses deterministic rather than stochastic estimates of the volume rendering integral.  DIVeR's representation is a voxel based field of features.  To compute the volume rendering integral, a ray is broken into intervals, one per voxel; components of the volume rendering integral are estimated from the features for each interval using an MLP, and the components are aggregated.  As a result, DIVeR can render thin translucent structures that are missed by other integrators.  Furthermore, DIVeR's representation has semantics that is relatively exposed compared to other such methods -- moving feature vectors around in the voxel space results in natural edits. Extensive qualitative and quantitative comparisons to current state-of-the-art methods show that DIVeR produces models that (1) render at or above state-of-the-art quality, (2) are very small without being baked, (3) render very fast without being baked, and (4) can be edited in natural ways.

## Results

|Ship|Lego|Drums|
|----|----|----|
|<img src="https://user-images.githubusercontent.com/93700419/175054047-4228ff06-8303-4267-9bba-e6a767763c5b.gif" width="512">|<img src="https://user-images.githubusercontent.com/93700419/175054062-28986580-28ef-4244-8e68-733e49bce249.gif" width="512">|<img src="https://user-images.githubusercontent.com/93700419/175054430-e904782a-270e-4e6f-8fee-2c8675b61c39.gif" width="512">|

## Citation
```
@misc{wu2021diver,
      title={DIVeR: Real-time and Accurate Neural Radiance Fields with Deterministic Integration for Volume Rendering}, 
      author={Liwen Wu and Jae Yong Lee and Anand Bhattad and Yuxiong Wang and David Forsyth},
      year={2021},
      eprint={2111.10427},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
