#!/bin/bash
#
# Grid Engine options (lines prefixed with #$)
#
# Set name of job
#$ -N osu-diffusion-training
#
# Set working directory to the directory where the job is submitted from:
#$ -cwd
#
# Request one GPU in the gpu queue:
#$ -q gpu
#$ -pe gpu-a100 2
#
# Request 96 GB system RAM
# the total system RAM available to the job is the value specified here multiplied by
# the number of requested GPUs (above)
#$ -l h_vmem=96G
#
# Log file
#$ -o logs/train.log
#$ -e logs/train.err
#
# Initialise the environment modules and load CUDA version 11.0.2
. /etc/profile.d/modules.sh
module load cuda
module load python/3.11.4
# Run the executable
export PATH=~/.local/bin:$PATH
export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'
export CURRENT_CHECKPOINT="results/005-DiT-L/checkpoints/0500000.pt"
while true
do
# torchrun --nproc-per-node=2 train.py --data-path "../ORS13402_no_audio" --model DiT-L --num-workers 1 --epochs 100 --global-batch-size 128 --ckpt-every 20000 --seq-len 128 --ckpt $CURRENT_CHECKPOINT
# Fallback to batchsize 64
torchrun --nproc-per-node=2 train.py --data-path "../ORS13402_no_audio" --model DiT-L --num-workers 1 --epochs 100 --global-batch-size 64 --ckpt-every 20000 --seq-len 128 --ckpt $CURRENT_CHECKPOINT
sleep 1
done
