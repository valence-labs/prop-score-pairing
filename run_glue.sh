#!/bin/bash
 
#SBATCH --job-name=matching       
#SBATCH --account=st-benbr-1-gpu    
#SBATCH --nodes=1                  
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4                    
#SBATCH --mem=32G                  
#SBATCH --time=5:00:00             
#SBATCH --gpus-per-node=1
#SBATCH --output=output.txt         
#SBATCH --error=error.txt          
#SBATCH --mail-user=johnny.xi@stat.ubc.ca
#SBATCH --mail-type=ALL                               
 
module load gcc/9.4.0 python/3.8.10 py-virtualenv/16.7.6
module load http_proxy

source ~/matching/bin/activate
 
cd $SLURM_SUBMIT_DIR

export NUMBA_CACHE_DIR='/tmp/'

python3 run_scglue.py




