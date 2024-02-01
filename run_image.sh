#!/bin/bash
 
#SBATCH --job-name=matching-array       
#SBATCH --account=st-benbr-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    
#SBATCH --mem=32G                  
#SBATCH --time=4:00:00             
#SBATCH --gpus-per-node=2
#SBATCH --output=outputs/matching_out_%A_%a.txt         
#SBATCH --error=outputs/matching_err_%A_%a.txt         
#SBATCH --mail-user=johnny.xi@stat.ubc.ca
#SBATCH --mail-type=ALL                               
 
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
module load http_proxy

source ~/matching/bin/activate
 
cd $SLURM_SUBMIT_DIR
export WANDB_DATA_DIR=$SLURM_SUBMIT_DIR/outputs

python3 main.py --max_epochs=100 --dataset=BALLS --model=CLASSIFIER --seed=$SLURM_ARRAY_TASK_ID --eval_interval=100
python3 main.py --max_epochs=100 --dataset=BALLS --model=VAE --seed=$SLURM_ARRAY_TASK_ID --eval_interval=100



