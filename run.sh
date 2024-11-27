#!/bin/bash

# Unload existing Anaconda modules
module unload anaconda3_cpu
module unload anaconda3_gpu

# Load the desired Anaconda module
module load anaconda3_gpu

# Deactivate any existing conda environment
conda deactivate

# Activate the base conda environment
conda activate base

# Create a new conda environment with Python 3.12
conda create -n cs_598_aie python=3.12 -y

# Activate the newly created environment
conda activate cs_598_aie

# Install the current directory as a Python package
pip install .
