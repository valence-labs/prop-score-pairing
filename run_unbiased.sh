#!/bin/bash
 
#SBATCH --job-name=probe-array-unbiased      
#SBATCH --account=st-benbr-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    
#SBATCH --mem=32G                  
#SBATCH --time=3:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=outputs/probing_out_%A_%a.txt         
#SBATCH --error=outputs/probing_err_%A_%a.txt           
#SBATCH --mail-user=johnny.xi@stat.ubc.ca
#SBATCH --mail-type=ALL                                  
 
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
module load http_proxy

source ~/matching/bin/activate
 
cd $SLURM_SUBMIT_DIR
export WANDB_DATA_DIR=$SLURM_SUBMIT_DIR/outputs

checkpoint_classifier=results/checkpoints/CLASSIFIERGEXADT$SLURM_ARRAY_TASK_ID-v1.ckpt
checkpoint_vae=results/checkpoints/VAEGEXADT$SLURM_ARRAY_TASK_ID-v1.ckpt



python3 probing.py --checkpoint=$checkpoint_classifier --max_epochs=50 --seed=$SLURM_ARRAY_TASK_ID --unbiased
python3 probing.py --checkpoint=$checkpoint_vae --max_epochs=50 --seed=$SLURM_ARRAY_TASK_ID --unbiased
python3 probing.py --checkpoint=random --max_epochs=50 --seed=$SLURM_ARRAY_TASK_ID --unbiased
python3 probing.py --checkpoint=gt --max_epochs=50 --seed=$SLURM_ARRAY_TASK_ID --unbiased



