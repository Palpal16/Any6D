#!/bin/bash

set -e

# create conda environment
conda create -n any6d python=3.9 -y

# activate conda environment
conda activate any6d

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"

# install dependencies from requirements
pip install -r requirements_clean.txt

# install specialized packages
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-deps kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121


# setup foundationpose
conda install cmake -y
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash foundationpose/build_all_conda.sh


# CUDA_VISIBLE_DEVICES=1 python run_demo.py --ycb_model_path demo_data/ --img_to_3d