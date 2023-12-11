#!/bin/bash 
 
#SBATCH --ntasks=1 
#SBATCH --mem=8000MB 
#SBATCH --time=96:00:00 
#SBATCH --job-name=example_run 
#SBATCH --output=outputs/example_run_out.txt 
 
source ~/.bashrc 
conda activate OzStar 
time python main.py configs/example.ini