#!/bin/bash

#SBATCH --account=jhjin1
#SBATCH --job-name=SIMCLR
#SBATCH --mail-user=xysong@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=16GB
#SBATCH --time=30:00:00
#SBATCH --output=/scratch/sunwbgt_root/sunwbgt98/xysong/IISE-QCRE-Data-Challenge/out.log

python src/simclr_pretrain.py 