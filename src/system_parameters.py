


import logging
logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)
import configparser
from pathlib import Path






def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent




"""
Class of parameters which define the system
"""
class SystemParameters:


    def __init__(self,config_file=None,
                 T       = 10,               # how long to integrate for in years
                 cadence = 7,                # the interval between observations in days
                 Ω       = 5.0e-7,           # GW angular frequency
                 Φ0      = 0.20,             # GW phase offset at t=0
                 ψ       =  2.50,            # GW polarisation angle
                 ι       = 1.0,              # GW source inclination
                 δ       =  1.0,             # GW source declination
                 α       =  1.0,             # GW source right ascension
                 h       =  5e-15,           # GW strain
                 process_noise = 'Fixed',    # the process noise on the pulsars. Any of "True", "Fixed", "Random". See pulsars.py for example
                 σm = 1e-11,                 # measurement noise standard deviation
                 Npsr = 0,                   # Number of pulsars to use in PTA. 0 = all
                 measurement_model='pulsar', # what do you want the KF measurement model to be? One of pulsar, earth,null
                 seed = 1234,                # this is the noise seed. It is used for realisations of process noise and measurement noise and also if random pulsars or random process noise covariances are requested 
                 σp = 1e-20,                  # only used if process_noise != True. Assign the process noise s.d. = σp for all pulsars if "Fixed". Assign randomly within U(σp/10,σp*10) if random. 
                 γ  = 1e-13                  # Mean reversion. The same for every pulsar ): 
                 ):
        logging.info("Welcome to the Kalman Filter Nested Sampler for PTA GW systems")

        #If a config file is passed, read from that preferentially.
        #Otherwise use the arguments
        #This lets us pass arguments to SystemParameters when exploring/testing, but also pass a config file for major runs
        #todo: unit tests for config parsing 


        if config_file is not None:
            #Read in the config file
            config = configparser.ConfigParser()
            config.read(config_file)

            #Assign variables to self
            #Could just leave them defined in the config - style choice

            #GW parameters
            GW = config['GW_PARAMETERS']

            self.Ω  = float(GW['Ω'])
            self.Φ0 = float(GW['Φ0'])
            self.ψ  = float(GW['ψ'])
            self.ι  = float(GW['ι'])
            self.δ  = float(GW['δ'])
            self.α  = float(GW['α'])
            self.h  = float(GW['h'])

            #PSR parameters
            PSR = config['PSR_PARAMETERS']

            self.process_noise = PSR['process_noise']
            self.Npsr          = int(PSR['Npsr'])
            self.σp            = float(PSR['σp'])
            self.γ             = float(PSR['γ'])
            self.PTA_file      = PSR['PTA_data_file']


            #OBS parameters
            OBS = config['OBS_PARAMETERS']

            self.T       = float(OBS['T'])
            self.cadence = float(OBS['cadence'])
            self.σm      = float(OBS['σm'])
            self.seed    = int(OBS['seed'])


            #INFERENCE parameters. Just the measurement model which is used by the KF
            #All other settings for the sampler are defined in NestedSamplerSettings
            INF = config['INFERENCE_PARAMETERS']
            self.measurement_model = INF['measurement_model']



        else:
            self.T = T 
            self.cadence = cadence
            self.Ω = Ω
            self.Φ0 = Φ0
            self.ψ = ψ
            self.ι = ι
            self.δ = δ
            self.α = α
            self.h = h
            self.σp = σp 
            self.process_noise = process_noise
            self.σm = σm
            self.Npsr = int(Npsr)

            self.measurement_model = measurement_model
            self.seed = seed
            self.γ = γ
            root = get_project_root()
            self.PTA_file = root / "data/NANOGrav_pulsars.csv" #hardcoded



        logging.info(f"Random seed is {self.seed}")









    
"""Class of parameters which define the settings used by the Bilby nested sampler
    Separate from SystemParameters for a) Organisation and b) I don't want to pass any true values to the sampler in any way 
"""
class NestedSamplerSettings:

        def __init__(self,config_file=None,
                     label='example',
                     outdir = '../data/nested_sampling/',
                     sampler = 'dynesty',
                     sample = 'rwalk_dynesty',
                     bound = 'multi',
                     dlogz=0.1,
                     npoints=1000,
                     npool=1,
                     plot=False,
                     resume=False
                     ): 

            logging.info("Getting the settings for nested sampling")

            if config_file is not None:
                #Read in the config file
                config = configparser.ConfigParser()
                config.read(config_file)
                logging.info("NS settings from config file")

                INF = config['INFERENCE_PARAMETERS']
                self.label   = INF['label']
                self.outdir  = INF['outdir']
                self.sampler = INF['sampler']
                self.sample  = INF['sample']
                self.bound   = INF['bound']
                self.dlogz   = float(INF['dlogz'])
                self.npoints = int(INF['npoints'])
                self.npool   = int(INF['npool'])
                self.plot    = eval(INF['plot']) #cast to boolean
                self.resume  = eval(INF['resume']) #cast to boolean

            else: #just assign canonical values. Almost never used, just here for consistency
                self.label   = label
                self.outdir  = outdir
                self.sampler = sampler
                self.sample  = sample
                self.bound   = bound
                self.dlogz   = dlogz
                self.npoints = npoints
                self.npool   = npool
                self.plot    = plot
                self.resume  = resume