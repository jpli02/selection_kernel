#!/bin/bash
#SBATCH --job-name=TwoPassAccum
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --account=bcjw-delta-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --output=eval.log
#SBATCH --output=two_pass_accum_output_%j.log
#SBATCH --error=two_pass_accum_error_%j.log
#SBATCH --mem-per-cpu=20G  # Adjusted memory to prevent OOM error

module load cuda/12.4.0

export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/u/ndani/.conda/cs598/lib:$LD_LIBRARY_PATH

source ~/.bashrc
conda deactivate
conda activate aie

python3 tests.py --Z 2 --H 2 --N_CTX 128 --HEAD_DIM 128
python3 tests.py --Z 2 --H 2 --N_CTX 2048 --HEAD_DIM 128
python3 tests.py --Z 4 --H 4 --N_CTX 128 --HEAD_DIM 256
python3 tests.py --Z 8 --H 2 --N_CTX 1024 --HEAD_DIM 128
python3 tests.py --Z 16 --H 16 --N_CTX 1024 --HEAD_DIM 128
python3 tests.py --Z 8 --H 2 --N_CTX 1024 --HEAD_DIM 128
python3 tests.py --Z 16 --H 16 --N_CTX 8096 --HEAD_DIM 4096
