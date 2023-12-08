import configparser

# A script for constructing a config file to be read by the files in `src`
# This should enforce reproducibility - no free parameters in `src`

config = configparser.ConfigParser()
config.optionxform = lambda option: option #enforce case sensitivity

config['GW_PARAMETERS'] = {'Ω':  5e-7, # GW angular frequency
                           'Φ0': 0.20,# GW phase offset at t=0
                           'ψ':  2.50, # GW polarisation angle
                           'ι':  1.0,  # GW source inclination
                           'δ':  1.0,  # GW source declination
                           'α':  1.0,  # GW source right ascension
                           'h':  5e-15 # GW strain
                           }

config['PSR_PARAMETERS'] = {'process_noise': 'Fixed', # the process noise on the pulsars. Any of "True", "Fixed", "Random". See pulsars.py for example
                            'Npsr': 0,                # Number of pulsars to use in PTA. 0 = all
                            'σp': 1e-20,              # only used if process_noise != True. Assign the process noise s.d. = σp for all pulsars if "Fixed". Assign randomly within U(σp/10,σp*10) if random. 
                            'γ': 1e-13,               # mean reversion. the same for every pulsar
                            'PTA_data_file': "../data/NANOGrav_pulsars.csv"
                            } 


config['OBS_PARAMETERS'] = {'T': 10,       # how long to integrate for in years
                            'cadence': 7,  # the interval between observations in days
                            'σm':1e-11,    # measurement noise standard deviation
                            'seed':1,      # this is the noise seed. It is used for realisations of process noise and measurement noise and also if random pulsars or random process noise covariances are requested 
                             }



config['INFERENCE_PARAMETERS'] = {'measurement_model': 'pulsar',        # what do you want the KF measurement model to be? One of pulsar, earth,null
                       'label': 'example_run',               # name of the run 
                       'outdir': "../data/nested_sampling/", # where to store the run output
                       'sampler': 'dynesty',                 # sampler to use
                       'sample': 'rwalk_dynesty',            # sampling method
                       'bound': 'multi',                     # bounding method. Other options include 'single', 'auto'. See https://dynesty.readthedocs.io/en/latest/faq.html
                       'dlogz': 0.1,                         # termination criteria
                       'npoints':1000,                       # number of live points
                       'npool': 1,                           # number of parallel threads
                       'plot': False,                        # do you want to plot the results?
                       'resume':False                        # do you want to resume from an earlier run using an existing pickle file?
                      }


with open('example.ini', 'w') as configfile:
  config.write(configfile)