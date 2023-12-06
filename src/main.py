


import logging 
from system_parameters import SystemParameters
from pulsars import Pulsars
from synthetic_data import SyntheticData
from model import PhaseModel
from kalman_filter import KalmanFilter
from priors import bilby_priors_dict
from bilby_wrapper import BilbySampler

import sys 
import time 





def bilby_inference_run(arg_name):
    logger = logging.getLogger().setLevel(logging.INFO)
    
    #Setup and create some synthetic data
    P   = SystemParameters(seed=1230,Npsr=2,σm=5e-7)    # User-specifed system parameters
    PTA = Pulsars(P)            # All pulsar-related quantities
    data = SyntheticData(PTA,P) # Given the user parameters and the PTA configuration, create some synthetic data
    
    #Define the model to be used by the Kalman Filter
    model = PhaseModel(P)
    
    
    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.phi_measured,PTA)

    #Run the KF with the correct parameters.
    #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
    init_parameters,optimal_parameters_dict = bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters = optimal_parameters_dict.sample(1)    
    model_likelihood,xresults,yresults = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")



    # #Run it again to profile
    # with Profile() as profile:
    #     model_likelihood,xresults,yresults = KF.likelihood(optimal_parameters)
    #     stats = Stats(profile)
    #     stats.sort_stats('tottime').print_stats(50)

    

    
    # #Bilby
    # init_parameters, priors = bilby_priors_dict(PTA,P)
   

    # logging.info("Testing KF using parameters sampled from prior")
    # params = priors.sample(1)
    # model_likelihood,xresults,yresults = KF.likelihood(params)
    # logging.info(f"Non -ideal likelihood for randomly sampled parameters = {model_likelihood}")

    
    # #Now run the Bilby sampler
    # BilbySampler(KF,init_parameters,priors,label=arg_name,outdir="../data/nested_sampling/",npoints=1000)
    # logging.info("The run has completed OK")



from cProfile import Profile
from pstats import Stats
arg_name = sys.argv[1]           # reference name
if __name__=="__main__":

    
    bilby_inference_run(arg_name)


    #profile.print_stats(sort='time')






