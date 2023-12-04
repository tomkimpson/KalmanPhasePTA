



# from src import main


# """
# Test that the general workflow runs before we call Bilby sampler 
# """
# def test_main():


#     #Setup and create some synthetic data
#     P   = SystemParameters(seed=1230,Npsr=0,σm=5e-7)    # User-specifed system parameters
#     PTA = Pulsars(P)            # All pulsar-related quantities
#     data = SyntheticData(PTA,P) # Given the user parameters and the PTA configuration, create some synthetic data
    
#     #Define the model to be used by the Kalman Filter
#     model = PhaseModel(P)
    
    
#     #Initialise the Kalman filter
#     KF = KalmanFilter(model,data.phi_measured,PTA)

#     #Run the KF with the correct parameters.
#     #We get the correct parameters via Bilby dictionary, looking towards when we will run this with nested sampling
#     init_parameters,optimal_parameters_dict = bilby_priors_dict(PTA,P,set_state_parameters_as_known=True,set_measurement_parameters_as_known=True)
#     optimal_parameters = optimal_parameters_dict.sample(1)    
#     model_likelihood,xresults,yresults = KF.likelihood(optimal_parameters)
#     logging.info(f"Ideal likelihood given optimal parameters = {model_likelihood}")

