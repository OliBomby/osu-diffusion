#!/bin/bash
#
# Grid Engine options (lines prefixed with #$)
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 4
#
# Request 4 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=8G
# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda
# Run the executable
torchrun --nproc-per-node=4 python train.py --data-path "../ORS13402_no_audio" --model DiT-XL --num-workers 12 --epochs 100 --global-batch-size 256 --ckpt-every 20000 --seq-len 128