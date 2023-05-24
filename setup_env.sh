#!/bin/bash

# Set environment variables
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"

# Create and activate new conda environment
conda create --name pytorch1.13
conda activate pytorch1.13

# Intall Pytorch, TorchVision, PyG and Shapely
conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge shapely

# Deactivate and reactivate environment such that correct python path is used
conda deactivate
conda activate pytorch1.13

# Install MMDetection
pip install -U openmim
mim install mmengine
mim install mmcv
python -m pip install git+https://github.com/open-mmlab/mmdetection.git

# Install from git repositories using pip
python -m pip install git+https://github.com/CedricPicron/boundary-iou-api.git
python -m pip install git+https://github.com/CedricPicron/Deformable-DETR.git
python -m pip install git+https://github.com/facebookresearch/detectron2.git

# Install local extensions
cd models/extensions/deformable
./make.sh
cd -
