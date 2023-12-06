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
        
     

        
        F = model.F_function(P.γ, PTA.dt,PTA.Npsr)
        Q = model.Q_function(P.γ,P.σp,PTA.dt,PTA.Npsr)
        R = model.R_function(P.σm)


        assert F.shape == (2*PTA.Npsr, 2*PTA.Npsr) 
        assert Q.shape == F.shape

        #Floats have no shape so no shape test for R

"""Check the matrices reduced to what we expect in the zero case"""
def test_zero_values():

    Npsr = 10
    F = model.F_function(1e-13, 0.0,Npsr)
    assert np.all(F == np.eye(2*Npsr)) #when dt is zero, F is just an identity matrix


    Q = model.Q_function(1e-13,1e-3,0.0,Npsr)
    assert np.all(Q == np.zeros(2*Npsr)) #when sigma_p is zero, Q is all zeros



    R = model.R_function(0.0)
    assert np.all(R == np.zeros(Npsr)) #when dt is zero, F is just an identity matrix




