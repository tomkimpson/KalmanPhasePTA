import configparser
import numpy as np




def create_ini(config_path,h=5e-15,σm=2*np.pi*1e-5):

    config = configparser.ConfigParser()
    config.optionxform = lambda option: option #enforce case sensitivity

    config['GW_PARAMETERS'] = {'Ω':  5e-7, # GW angular frequency
                            'Φ0': 0.20,# GW phase offset at t=0
                            'ψ':  2.50, # GW polarisation angle
                            'ι':  1.0,  # GW source inclination
                            'δ':  1.0,  # GW source declination
                            'α':  1.0,  # GW source right ascension
                            'h':  h # GW strain
                            }

    config['PSR_PARAMETERS'] = {'process_noise': 'Fixed', # the process noise on the pulsars. Any of "True", "Fixed", "Random". See pulsars.py for example
                                'Npsr': 0,                # Number of pulsars to use in PTA. 0 = all
                                'σp': 1e-20,              # only used if process_noise != True. Assign the process noise s.d. = σp for all pulsars if "Fixed". Assign randomly within U(σp/10,σp*10) if random. 
                                'γ': 1e-13,               # mean reversion. the same for every pulsar
                                'PTA_data_file': "../data/NANOGrav_pulsars.csv"
                                } 


    config['OBS_PARAMETERS'] = {'T': 10,       # how long to integrate for in years
                                'cadence': 7,  # the interval between observations in days
                                'σm':σm,    # measurement noise standard deviation
                                'seed':1230,      # this is the noise seed. It is used for realisations of process noise and measurement noise and also if random pulsars or random process noise covariances are requested 
                                }



    config['INFERENCE_PARAMETERS'] = {'measurement_model': 'pulsar',        # what do you want the KF measurement model to be? One of pulsar, earth,null
                        'label': f'batch_{h}',               # name of the run 
                        'outdir': "../data/nested_sampling/", # where to store the run output
                        'sampler': 'dynesty',                 # sampler to use
                        'sample': 'rwalk_dynesty',            # sampling method
                        'bound': 'live',                     # bounding method. Other options include 'single', 'auto'. See https://dynesty.readthedocs.io/en/latest/faq.html
                        'dlogz': 0.1,                         # termination criteria
                        'npoints':1000,                       # number of live points
                        'npool': 1,                           # number of parallel threads
                        'plot': False,                        # do you want to plot the results?
                        'resume':False                        # do you want to resume from an earlier run using an existing pickle file?
                        }



    with open(config_path, 'w') as configfile:
        config.write(configfile)




def create_slurm_job(config_file):


    config = configparser.ConfigParser()
    config.read(config_file)
    arg_name = config['INFERENCE_PARAMETERS']['label']

    with open(f'slurm_jobs/slurm_{arg_name}.sh','w') as g:
        g.write("#!/bin/bash \n \n")  
        g.write("#SBATCH --ntasks=1 \n")  
        g.write("#SBATCH --mem=8000MB \n")  
        g.write("#SBATCH --time=96:00:00 \n")  
        g.write(f"#SBATCH --job-name={arg_name} \n")  
        g.write(f"#SBATCH --output=outputs/{arg_name}_out.txt \n \n")

        g.write("source ~/.bashrc \n")
        g.write("conda activate OzStar \n")
        g.write(f"time python main.py {config_file}")

    return arg_name
        

#First create the ini file 
hvals = [1e-12,1e-13,1e-14,1e-15]
with open('batch.sh','w') as b: 
    for h in hvals:
        #Create the ini file
        config_path = f'configs/batch_{h}.ini'
        create_ini(config_path,h=h,σm=2*np.pi*1e-5)

        #And the slurm job for the ini file
        arg_name = create_slurm_job(config_path)
        b.write(f"sbatch slurm_jobs/slurm_{arg_name}.sh & \n")