


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


# This is a profiling script used to see how long individual likelihood evaluatins take
# It is analogous to main.py, but without the Bilby Bayesian inference
# The likelihood runtime is a key constraint on how long the sampler takes, so we want to get this as small as possible


def bilby_inference_run():
    logger = logging.getLogger().setLevel(logging.INFO)
    
    #Setup and create some synthetic data
    P   = SystemParameters(seed=1230,Npsr=2,σm=5e-7)    # User-specifed system parameters
    PTA = Pulsars(P)            # All pulsar-related quantities
    data = SyntheticData(PTA,P) # Given the user parameters and the PTA configuration, create some synthetic data
    
    #Define the model to be used by the Kalman Filter
    model = PhaseModel(P,PTA)
    
    
    #Initialise the Kalman filter
    KF = KalmanFilter(model,data.phi_measured,PTA)

    #Run the KF with the correct parameters.
    #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
    init_parameters,optimal_parameters_dict = bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters = optimal_parameters_dict.sample(1)    
    model_likelihood = KF.likelihood(optimal_parameters)
    logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")




    #First time it naively
    t0 = time.time()
    model_likelihood = KF.likelihood(optimal_parameters)
    t1=time.time()
    print("Runtime = ", t1-t0)


    #Run it again to profile
    from cProfile import Profile
    from pstats import Stats
    with Profile() as profile:
        model_likelihood = KF.likelihood(optimal_parameters)
        stats = Stats(profile)
        stats.sort_stats('tottime').print_stats(10)

    
    print ("Check likelihood is reasonable = ", model_likelihood)
   

if __name__=="__main__":
    bilby_inference_run()






