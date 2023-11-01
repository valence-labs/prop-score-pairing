#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=5:00:00
#SBATCH --mem-per-gpu=100G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate multiomics

srun python run_scglue.py


