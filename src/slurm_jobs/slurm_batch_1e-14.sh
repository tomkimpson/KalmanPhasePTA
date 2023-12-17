#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=96:00:00 
#SBATCH --job-name=batch_1e-14 
#SBATCH --output=outputs/batch_1e-14_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py configs/batch_1e-14.ini