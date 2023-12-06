
from src import system_parameters,pulsars,synthetic_data,model,priors,kalman_filter
from test import kalman_filter_naive as KF_naive
import numpy as np 
"""We make a few optimisations for speed. Lets check that these hold.
We use a naive version of the KF, test/kalman_filter_naive.py
This is just all the usual canonical definitions"""
def test_optimisations():

    #Setup and create some synthetic data
    P   = system_parameters.SystemParameters(seed=1230,σm=5e-7)    
    PTA = pulsars.Pulsars(P)            
    data = synthetic_data.SyntheticData(PTA,P) 
    
    #Define the model to be used by the Kalman Filter
    phase_model = model.PhaseModel(P)

    #Parameters
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters = optimal_parameters_dict.sample(1)    
    
    
    #Initialise the naive Kalman filter
    KF = KF_naive.KalmanFilter(phase_model,data.phi_measured,PTA)
    model_likelihood_naive,xresults,yresults = KF.likelihood(optimal_parameters)

    #Now do the acutal KF
    KF = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)
    model_likelihood_actual,xresults,yresults = KF.likelihood(optimal_parameters)


    assert model_likelihood_actual == model_likelihood_naive


    #also check the sub-optimal parameters

    #Parameters
    init_parameters,sub_optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False)
    sub_optimal_parameters = sub_optimal_parameters_dict.sample(1)    
    
    
    #Initialise the naive Kalman filter
    KF = KF_naive.KalmanFilter(phase_model,data.phi_measured,PTA)
    model_likelihood_naive,xresults,yresults = KF.likelihood(sub_optimal_parameters)

    #Now do the acutal KF
    KF = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)
    model_likelihood_actual,xresults,yresults = KF.likelihood(sub_optimal_parameters)
    
    np.testing.assert_approx_equal(model_likelihood_actual, model_likelihood_naive) #approx c.f. machine precision
