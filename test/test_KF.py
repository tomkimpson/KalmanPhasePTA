from src import system_parameters,pulsars,synthetic_data,model, kalman_filter,kalman_filter_optimised, priors
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

"""Also check that the optimised likelihood returns the same as the standard likelihood """

def test_optimised_likelihoods():

   
    P   = system_parameters.SystemParameters()    # User-specifed system parameters
    PTA = pulsars.Pulsars(P)
    data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

    phase_model = model.PhaseModel(P,PTA)
    
    #Initialise the Kalman filter
    KF_standard = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)
    KF_opt = kalman_filter_optimised.KalmanFilter(phase_model,data.phi_measured,PTA)

    #Run the KF with the correct parameters.
    #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
    init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
    optimal_parameters = optimal_parameters_dict.sample(1) 


    ll_standard = KF_standard.likelihood(optimal_parameters)
    ll_opt = KF_opt.likelihood(optimal_parameters)
    assert ll_standard == ll_opt

    #also check that the wrong parameters give the same likelihood
    init_parameters,sub_optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False)
    sub_optimal_parameters = sub_optimal_parameters_dict.sample(1)    
    ll_standard = KF_standard.likelihood(sub_optimal_parameters)
    ll_opt = KF_opt.likelihood(sub_optimal_parameters)
    assert ll_standard == ll_opt




    #Now try across some different parameters

    N = 5
    Ω = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    Φ0= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ψ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ι= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    δ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    α= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    h= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    T = np.random.uniform(1,10,size=N) #also vary obs time

    
    for i in range(N):
        P   = system_parameters.SystemParameters(Ω=Ω[i],Φ0=Φ0[i],ψ=ψ[i],ι=ι[i],δ=δ[i],α=α[i],h=h[i],T=T[i]) 
        PTA = pulsars.Pulsars(P)
        data = synthetic_data.SyntheticData(PTA,P)            # Given the system parameters and the PTA configuration, create some synthetic data

        phase_model = model.PhaseModel(P,PTA)
        
        #Initialise the Kalman filter
        KF_standard = kalman_filter.KalmanFilter(phase_model,data.phi_measured,PTA)
        KF_opt = kalman_filter_optimised.KalmanFilter(phase_model,data.phi_measured,PTA)

        #Run the KF with the correct parameters.
        #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
        init_parameters,optimal_parameters_dict = priors.bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
        optimal_parameters = optimal_parameters_dict.sample(1) 


        ll_standard = KF_standard.likelihood(optimal_parameters)
        ll_opt = KF_opt.likelihood(optimal_parameters)
        np.testing.assert_almost_equal(ll_standard,ll_opt,decimal=5) #some small differences expected due to finite float precision