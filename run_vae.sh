#!/bin/bash

#SBATCH --job-name=matching_vae
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=20G


eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun python train_vae.py --dataset=GEXADT --max_epochs=500


