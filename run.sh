#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem-per-gpu=100G
#SBATCH --output=/results/slurm-%A.out


eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

srun python -W ignore::RuntimeWarning main.py --dataset=GEXADT --lr=0.0005 --max_epochs=100 --batch_size=256
srun python -W ignore::RuntimeWarning main.py --dataset=BALLS --lr=0.0001 --max_epochs=100 --batch_size=50

pyenv activate multiomics

srun python -W run_scglue.py --dataset=BALLS --lr=0.0001 --max_epochs=100 --batch_size=50
