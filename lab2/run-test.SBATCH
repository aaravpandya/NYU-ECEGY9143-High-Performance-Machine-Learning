#!/bin/bash

#SBATCH --time=2:00:00
#SBATCH --mem=12GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch
#SBATCH --account=ece_gy_9143-2023sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --output=%x.out

singularity exec --nv \
            --overlay /scratch/ap7641/pytorch.ext3:ro \
            /share/apps/images/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
            /bin/bash -c "source /ext3/env.sh; python index.py --question c2 --epochs=5;python index.py --question c3;python index.py --question c4;python index.py --question c5 --epochs 5;python index.py --question c6 --epochs=5;"
