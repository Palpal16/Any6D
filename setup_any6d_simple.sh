#!/bin/bash

set -e

echo "=== Any6D Setup Semplificato ==="

# create conda environment
conda create -n any6d python=3.9 -y

# activate conda environment
conda activate any6d

# Install system dependencies
sudo apt-get update -q
sudo apt-get install -y build-essential cmake libeigen3-dev

# Install Eigen3 3.4.0 under conda environment
conda install conda-forge::eigen=3.4.0 -y
export CMAKE_PREFIX_PATH="$CONDA_PREFIX/include/eigen3:$CMAKE_PREFIX_PATH"

# install dependencies from requirements
pip install -r requirements_clean.txt

# install specialized packages
pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-deps kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder pytorch3d==0.7.8+pt2.4.1cu121

# setup foundationpose
cd foundationpose
CMAKE_DIR=$(python -c "import pybind11; print(pybind11.get_cmake_dir())")
export CMAKE_PREFIX_PATH="$CMAKE_DIR:$CMAKE_PREFIX_PATH"
bash build_all_conda.sh
cd ..

# setup sam2
cd sam2
pip install -e .
cd ..

# setup instantmesh
cd instantmesh
pip install -r requirements.txt
cd ..

# setup bop_toolkit
cd bop_toolkit
pip install -e .
cd ..

echo "=== Setup completato! ==="
