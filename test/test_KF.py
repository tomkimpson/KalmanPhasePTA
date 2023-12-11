from src import system_parameters,pulsars,synthetic_data,model, kalman_filter, priors
# import random 
import numpy as np 
# from numpy import sin,cos

"""We have two likelihood functions, one for plotting. Lets double check they both give the same likelihoods"""
def test_both_likelihoods():



    for Npsr in [0,10,17]: #for some different PTAs
        P   = system_parameters.SystemParameters(Npsr=Npsr)    # User-specifed system parameters
        PTA = pulsars.Pulsars(P)
        data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

        phase_model = model.PhaseModel(P,PTA)
     
        #Initialise the Kalman filter
        KF = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)

        #Run the KF with the correct parameters.
        #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
        init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
        optimal_parameters = optimal_parameters_dict.sample(1)    
        model_likelihood1 = KF.likelihood(optimal_parameters)
        model_likelihood2,xx,yy = KF.likelihood_plotter(optimal_parameters)


        assert model_likelihood1 == model_likelihood2



        #also check that the wrong parameters give the same likelihood

        init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False)
        optimal_parameters = optimal_parameters_dict.sample(1)    
        model_likelihood1 = KF.likelihood(optimal_parameters)
        model_likelihood2,xx,yy = KF.likelihood_plotter(optimal_parameters)


        assert model_likelihood1 == model_likelihood2

