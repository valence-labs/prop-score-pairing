#!/bin/bash
 
#SBATCH --job-name=matching-array       
#SBATCH --account=st-benbr-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    
#SBATCH --mem=32G                  
#SBATCH --time=1:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=outputs/inference_out_%A_%a.txt         
#SBATCH --error=outputs/inference_err_%A_%a.txt         
#SBATCH --mail-user=johnny.xi@stat.ubc.ca
#SBATCH --mail-type=ALL                               
 
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
module load http_proxy

source ~/matching/bin/activate
 
cd $SLURM_SUBMIT_DIR

checkpoint_classifier=results/checkpoints/VAEBALLS$SLURM_ARRAY_TASK_ID.ckpt
checkpoint_vae=results/checkpoints/CLASSIFIERBALLS$SLURM_ARRAY_TASK_ID.ckpt

python3 inference.py --checkpoint=$checkpoint_classifier --dataset=BALLS
python3 inference.py --checkpoint=$checkpoint_vae --dataset=BALLS
