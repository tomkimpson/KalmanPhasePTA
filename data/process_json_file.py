
import json 
import pandas as pd
import numpy as np 
import glob

def process_results_file(path):

    print(f"Loading results from {path}")
    
    # Load the json results file
    f = open(path)
    data = json.load(f)
    df_posterior = pd.DataFrame(data["posterior"]["content"]) # posterior
    evidence = data["log_evidence"]
    f.close()

    variables_to_plot = ["omega_gw","phi0_gw","psi_gw","iota_gw","delta_gw","alpha_gw", "h"]

    #Create a numpy array of the variables you want to plot
    y_post = df_posterior[variables_to_plot].to_numpy()


    seed = p.split('_')[3]

    np.save(f'seed_{seed}_canonical_1',y_post)

list_of_files = sorted(glob.glob('*.json'))
for p in list_of_files:
    process_results_file(p)




