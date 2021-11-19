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

Edit `configs/config.py` to configure a training and setup dataset path.

To reproduce the results of the paper, replace `config.py` with other configuration files under the same folder.

The 'implicit' training stage takes around 40GB GPU memory and the 'implicit-explicit' stage takes around 20GB GPU memory. Decreasing the voxel grid size by a factor of 2 results in models that require around 10GB GPU memory, which causes acceptable deduction on rendering quality.

### Training

To train an explicit or implicit model:

```shell
python train.py --experiment_name=EXPERIMENT_NAME \
				--device=GPU_DEVICE\
				--resume=True # if want to resume a training
```

After training an implicit model, the explicit model can be trained:

```sh
python train.py --experiment_name=EXPERIMENT_NAME \
				--ft=CHECKPOINT_PATH_TO_IMPLICIT_MODEL_CHECKPOINT\
				--device=GPU_DEVICE\
				--resume=True
```

### Post processing

After the coarse model training and the fine 'implicit-explicit' model training, we perform voxel culling:

```shell
python prune.py --checkpoint_path=PATH_TO_MODEL_CHECKPOINT_FOLDER\
				--coarse_size=COARSE_IMAGE_SIZE\
				--fine_size=FINE_IMAGE_SIZE\
				--fine_ray=1 # to get rays that pass through non-empty space, 0 otherwise\
				--batch=BATCH_SIZE\
				--device=GPU_DEVICE
```

which stores the max-scattered 3D alpha map under model checkpoint folder as `alpha_map.pt` .  The rays that pass through non-empty space is also stored under model checkpoint folder. For Nerf-synthetic dataset, we directly store the rays in `fine_rays.npz`; for Tanks&Temples and BlendedMVS, we store the mask for each pixel under folder `masks` which indicates the pixels (rays) to be sampled.

To convert the checkpoint file in training to pytorch model weight or serialized weight file for real-time rendering:

```shell
python convert.py --checkpoint_path=PATH_TO_MODEL_CHECKPOINT_FILE\
				  --serialize=1 # if want to build serialized weight, 0 otherwise
```

The converted files will be stored under the same folder as the checkpoint file, where the pytorch model weight file is named as `weight.pth`, and the serialized weight file is named as `serialized.pth`

### Evaluation

To extract the offline rendered images:

```shell
python eval.py --checkpoint_path=PATH_TO_MODEL_CHECKPOINT_FILE\
			   --output_path=PATH_TO_OUTPUT_IMAGES_FOLDER\
			   --batch=BATCH_SIZE\
			   --device=GPU_DEVICE
```

To extract the real-time rendered images and test the mean FPS on the test sequence:

```sh
pyrhon eval_rt.py --checkpoint_path=PATH_TO_SERIALIZED_WEIGHT_FILE
				  --output_path=PATH_TO_OUPUT_IMAGES_FOLDER\
				  --decoder={32,64} # diver32, diver64\ 
				  --device=GPU_DEVICE
```

## Resources

- [Project page](https://lwwu2.github.io/diver/)
- Paper
- [Real-time Code](https://github.com/lwwu2/diver-rt)

## Citation

