


import logging
logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)
import configparser

"""
Class of parameters which define the system
"""
class SystemParameters:


    def __init__(self,config_file): 

        logging.info("Welcome to the Kalman Filter Nested Sampler for PTA GW systems")

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



        logging.info(f"Random seed is {self.seed}")


        
    
"""Class of parameters which define the settings used by the Bilby nested sampler
    Separate from SystemParameters for a) Organisation and b) I don't want to pass any true values to the sampler in any way 
"""
class NestedSamplerSettings:

        def __init__(self,config_file): 

            logging.info("Getting the settings for nested sampling")

            #Read in the config file
            config = configparser.ConfigParser()
            config.read(config_file)

            INF = config['INFERENCE_PARAMETERS']
            self.label   = INF['label']
            self.outdir  = INF['outdir']
            self.sampler = INF['sampler']
            self.sample  = INF['sample']
            self.bound   = INF['bound']
            self.dlogz   = float(INF['dlogz'])
            self.npoints = int(INF['npoints'])
            self.npool   = int(INF['npool'])
            self.plot    = bool(INF['plot'])
            self.resume  = bool(INF['resume'])