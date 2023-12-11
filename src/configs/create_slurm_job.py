

import sys
import configparser

### A python script that accepts a config file path and populates a slurm file 
config_file = sys.argv[1]
config = configparser.ConfigParser()
config.read(config_file)
arg_name = config['INFERENCE_PARAMETERS']['label']

with open('slurm.sh','w') as g:

    g.write("#!/bin/bash \n \n")  
    g.write("#SBATCH --ntasks=1 \n")  
    g.write("#SBATCH --mem=8000MB \n")  
    g.write("#SBATCH --time=96:00:00 \n")  
    g.write(f"#SBATCH --job-name={arg_name} \n")  
    g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

    g.write("source ~/.bashrc \n")
    g.write("conda activate OzStar \n")
    g.write(f"time python main.py {config_file}")
    
    
    