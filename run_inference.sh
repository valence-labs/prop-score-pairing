#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-gpu=100G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun python3 inference.py --checkpoint=results/checkpoints/PS-epoch=06-full_val_loss=0.63.ckpt --dataset=GEXADT --model=CLASSIFIER
srun python3 inference.py --checkpoint=results/checkpoints/PS-epoch=12-full_val_loss=0.16.ckpt --dataset=BALLS --model=CLASSIFIER
srun python3 inference.py --checkpoint=results/checkpoints/VAE-epoch=239-full_val_loss=0.47.ckpt --dataset=GEXADT --model=VAE
srun python3 inference.py --checkpoint=results/checkpoints/VAE-epoch=61-full_val_loss=820.31.ckpt --dataset=BALLS --model=VAE
