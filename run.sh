#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=20G


eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate multiomics

srun python run_scglue.py 

pyenv activate matching

srun python -W ignore::RuntimeWarning main.py --dataset=GEXADT --lr=0.0005 --max_epochs=100 --batch_size=256
srun python -W ignore::RuntimeWarning main.py --dataset=BALLS --lr=0.0001 --max_epochs=100 --batch_size=50


