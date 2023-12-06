#This is the test file for py_src/gravitational_waves.py 
from src import system_parameters,pulsars,synthetic_data,model
# import random 
import numpy as np 
# from numpy import sin,cos

"""Check the shapes of the matrices are as expected, since we construct these"""
def test_shapes_of_update_matrices():
    for Npsr in [0,10,17]: #for some different PTAs


        P   = system_parameters.SystemParameters(Npsr=Npsr)    # User-specifed system parameters
        PTA = pulsars.Pulsars(P)
        phase_model = model.PhaseModel(P,PTA)
     
        F,Q = phase_model.kalman_machinery()
             

        assert F.shape == (2*PTA.Npsr, 2*PTA.Npsr) 
        assert Q.shape == F.shape



"""Check the matrices reduced to what we expect in the zero case"""
def test_zero_values():


        #Check F becomes and identity matrix
        P   = system_parameters.SystemParameters()    # User-specifed system parameters
        PTA = pulsars.Pulsars(P)
        PTA.dt = 0.0 #modify dt to check outputs
        phase_model = model.PhaseModel(P,PTA)
        F,Q = phase_model.kalman_machinery()

        assert np.all(F == np.eye(2*PTA.Npsr)) #when dt is zero, F is just an identity matrix


        #Check Q is NOT zeros for σp = 0
        #This is because Q is defined a priori and does not depend on parameters to be inferred 
        #In kalman_filter Q is updated with σp^2
        P   = system_parameters.SystemParameters(σp=0.0)    # User-specifed system parameters
        PTA = pulsars.Pulsars(P)
        phase_model = model.PhaseModel(P,PTA)
        F,Q = phase_model.kalman_machinery()

        assert ~np.all(Q == np.zeros(2*PTA.Npsr)) #Q should not be a zeros array, but have non zero cpts





