#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=96:00:00 
#SBATCH --job-name=batch_1e-12 
#SBATCH --output=outputs/batch_1e-12_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py configs/batch_1e-12.ini