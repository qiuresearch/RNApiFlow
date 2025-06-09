#!/bin/bash

# Create a new conda environment
conda create -n RNAFlow python=3.10 -y
source activate RNAFlow

# Install core dependencies from conda-forge
conda install -c conda-forge numpy pandas gputil hydra-core  beartype jaxtyping dm-tree einops biopython loguru lightning wandb -y

conda install -c conda-forge pytorch_scatter pytorch_cluster pytorch_geometric -y

# Install PyTorch with CUDA support (adjust CUDA version as needed)
# cuda=12.4 12.6
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 05/27/2025: executing the commands above leads to erors but gives a working environment



# Install commonly problematic packages through conda
# conda install rich requests boto3 protobuf pyzmq wandb -y
# conda install uvicorn fastapi pydantic -y

# Try to install remaining pip packages
# pip install -r requirements.txt || echo "Some packages failed to install. "

echo "Conda environment setup complete! Some packages may need manual installation."