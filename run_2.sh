#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=20G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun python -W ignore::RuntimeWarning main.py --dataset=BALLS --max_epochs=250 --batch_size=100