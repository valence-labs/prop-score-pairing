#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-gpu=100G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun python -W ignore::RuntimeWarning main.py --dataset=GEXADT --lr=0.001 --max_epochs=200 --batch_size=256
srun python -W ignore::RuntimeWarning main.py --dataset=BALLS --lr=0.0001 --max_epochs=200 --batch_size=50