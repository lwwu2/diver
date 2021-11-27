# DIVeR: Deterministic Integration for Volume Rendering
This repo contains the training and evaluation code for DIVeR.

## Setup

- python 3.8
- pytorch 1.9.0
- pytorch-lightning 1.2.10
- torchvision 0.2.2
- torch-scatter 2.0.8

### Dataset

- [NeRF-Synthetic](https://github.com/bmild/nerf)
- [Tanks&Temples](https://github.com/facebookresearch/NSVF)
- [BlendedMVS](https://github.com/facebookresearch/NSVF)

### Pre-trained models

Both our real-time and offline models can be found in [here](https://drive.google.com/drive/folders/1iaQ-PIS7ydpoNj2OToLgAi-3lCyJG4MN?usp=sharing).

## Usage

Edit `configs/config.py` to configure a training.

### Training the coarse model

The coarse model is firstly trained on an explicit voxel grid and image resolution at 1/4 of the fine model scale with only a few epochs (5 in our experiment) to get the rough geometry (occupancy map):

```shell
python train.py --experiment_name=EXPERIMENT_NAME_COARSE \
                --device=GPU_DEVICE\
                --max_epochs=NUM_OF_EPOCHS
```

After the coarse model is trained, corresponding occupancy map is extracted and training rays are bias sampled to speed up the training of the fine model: 

```shell
python prune.py --checkpoint_path=PATH_TO_COARSE_MODEL_CHECKPOINT_FOLDER\
		--batch=BATCH_SIZE\
		--bias_sampling=1\ # 1 if bias sampling the training rays, 0 otherwise
		--device=GPU_DEVICE
```

The max-scattered 3D alpha map is stored under model checkpoint folder as `alpha_map.pt`.  The rays that pass through non-empty space are also stored under model checkpoint folder. For NeRF-synthetic dataset, we directly store the rays in `fine_rays.npz`; for Tanks&Temples and BlendedMVS, we store the mask for each pixel under folder `masks` which indicates the pixels (rays) to be sampled.

### Training the fine model

Given the coarse occupancy map and the bias sampled rays, an implicit fine model is trained to get rid of the overfitting:

```shell
python train.py --experiment_name=EXPERIMENT_NAME_IMPLICIT\
		--device=GPU_DEVICE
```

After the training curve is almost converged, the fine model is then trained at the 'implicit-explicit' stage:

```shell
python train.py --experiment_name=EXPERIMENT_NAME\
		--ft=CHECKPOINT_PATH_TO_IMPLICIT_MODEL\
		--device=GPU_DEVICE
```

Finally, The fine occupancy map is extracted:

```shell
python prune.py --checkpoint_path=PATH_TO_FINE_MODEL_CHECKPOINT_FOLDER\
		--batch=BATCH_SIZE\
		--device=GPU_DEVICE
```

Note: the 'implicit' training stage takes around 40GB GPU memory and the 'implicit-explicit' stage takes around 20GB GPU memory. Decreasing the voxel grid size by a factor of 2 (changing `voxel_num` and `mask_scale` in `config.py`) results in models that only require around 12GB GPU memory with acceptable deduction on rendering qualities.

### Conversion

To convert the checkpoint file in the training to pytorch state_dict or serialized weight file for real-time rendering:

```shell
python convert.py --checkpoint_path=PATH_TO_MODEL_CHECKPOINT_FILE\
		  --serialize={0,1} # 1 if want to build serialized weight, 0 otherwise
```

The converted files will be stored under the same folder as the checkpoint file, where the pytorch state_dict is named as `weight.pth`, and the serialized weight is named as `serialized.pth`

### Evaluation

To extract the offline rendered images:

```shell
python eval.py --checkpoint_path=PATH_TO_MODEL_CHECKPOINT_FILE\
	       --output_path=PATH_TO_OUTPUT_IMAGES_FOLDER\
	       --batch=BATCH_SIZE\
	       --device=GPU_DEVICE
```

To extract the real-time rendered images and test the mean FPS on the test sequence:

```shell
pyrhon eval_rt.py --checkpoint_path=PATH_TO_SERIALIZED_WEIGHT_FILE\
		  --output_path=PATH_TO_OUPUT_IMAGES_FOLDER\
		  --decoder={32,64} # diver32, diver64\ 
		  --device=GPU_DEVICE
```

### Reproduction

To reproduce the results of the paper, replace `config.py` with other configuration files under the same folder and change the `dataset_path` and the `coarse_path` (for fine model training). 

An example on the `drums` scene with `DIVeR32` model:
1. Replace `config.py` by  `nerf_synthetic_coarse.py`, set up `dataset_path` to `{DATASET_PATH}/drums`, and train the coarse model:
```shell
python train.py --experiment_name drums_coarse\ 
		--device 0\
		--max_epochs 5
```
2. Extract the coarse occupancy map: 
```shell
python prune.py --checkpoint_path checkpoints/drums_coarse\
		--batch 4000\
		--bias_sampling 1\
		--device 0
```
3. Replace `config.py` by `nerf_synthetic_fine_diver32.py`, set up `dataset_path`, set up `coarse_path` to `checkpoints/drums_coarse`, and train the fine implicit model: 
```shell
python train.py --experiment_name drums_im\
		--device 0
```
4. Train the 'implicit-explicit' model: 
```shell
python train.py --experiment_name drums\
		--ft checkpoints/drums_im/{BEST_MODEL}.ckpt\
		--device 0
```
and extract the fine occupancy map:
```shell
python prune.py --checkpoint_path checkpoints/drums\
		--batch 4000\
		--bias_sampling 0\
		--device 0
```
5. Convert training weight:
```shell
python convert.py --checkpoint_path checkpoints/drums/{BEST_MODEL}.ckpt\
		  --serialize 1
```
6. Offline rendering: 
```shell
python eval.py --checkpoint_path checkpoints/drums/weight.pth\
	       --output_path outputs/drums\
	       --batch 20480\
	       --device 0
``` 
7. Real-time rendering:
```shell
python eval_rt.py --checkpoint_path checkpoints/drums/serialize.pth\
		  --output_path outputs/drums\
		  --decoder 32\
		  --device 0
```

## Resources

- [Project page](https://lwwu2.github.io/diver/)
- [Paper](https://arxiv.org/abs/2111.10427)
- [Real-time Code](https://github.com/lwwu2/diver-rt)

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

