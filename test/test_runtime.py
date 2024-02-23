from cProfile import Profile
from pstats import Stats
from src import system_parameters,pulsars,synthetic_data,model, kalman_filter,kalman_filter_optimised, priors
import numpy as np 



"""Check the runtimes for a single likelihood evaluation. No assertions. Run with pytest -s for outputs """
def test_likelihood_runtimes():

   
    P   = system_parameters.SystemParameters()    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

    phase_model = model.PhaseModel(P,PTA)
    
    #Initialise the Kalman filter
    KF_standard = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)
    KF_opt = kalman_filter_optimised.KalmanFilter(phase_model,data.phi_measured,PTA)

    #Run once to allow numba precompile
    #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters = optimal_parameters_dict.sample(1) 
    ll_standard = KF_standard.likelihood(optimal_parameters)
    ll_opt = KF_opt.likelihood(optimal_parameters)

    print(ll_standard,ll_opt)

    #Run it again to profile
    with Profile() as profile:
        ll_standard = KF_standard.likelihood(optimal_parameters)
        stats = Stats(profile)
        stats.sort_stats('tottime').print_stats(10)

    
    with Profile() as profile:
        ll_standard = KF_opt.likelihood(optimal_parameters)
        stats = Stats(profile)
        stats.sort_stats('tottime').print_stats(10)

