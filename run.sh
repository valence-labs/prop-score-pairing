#!/bin/bash

#SBATCH --job-name=matching
#SBATCH --cpus-per-gpu=8
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=20G

eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"

pyenv activate matching

python3 -W ignore probing.py --max_epochs=250 --unbiased --lr=0.00002
python3 -W ignore probing.py --max_epochs=250 --lr=0.00002

