

import numpy as np
from numba import njit

import logging 
from gravitational_waves import gw_psr_terms,gw_earth_terms,null_model
import sys
from utils import block_diag_view_jit
class PhaseModel:

    """
    A linear model of the state evolution x = (\phi,f)
    
    """

    def __init__(self,P):

        """
        Initialize the class. 
        """
        if P.measurement_model == "null":
            logging.info("You are using just the null measurement model")
            self.H_function = null_model 
        elif P.measurement_model == "earth":
            logging.info("You are using the Earth terms measurement model")
            self.H_function = gw_earth_terms
        elif P.measurement_model == "pulsar":
            logging.info("You are using the Pulsar terms measurement model")
            self.H_function = gw_psr_terms
        else:
            sys.exit("Measurement model not recognized. Stopping.")








#These functions have to be outside the class to enable JIT compilation
#Bit ugly, but works from a performance standpoint
#Functions are as defined in docs/notes/TomModel_Scaled
"""
F matrix 
"""
@njit(fastmath=True)
def F_function(γ,dt,Npsr):
    component_array = np.array([[1,(1-np.exp(-γ*dt))/γ],
                                [0,np.exp(-γ*dt)]]) 

    return block_diag_view_jit(component_array,Npsr) 

"""
Q matrix
"""
@njit(fastmath=True)
def Q_function(γ,σp,dt,Npsr):
    #TODO this assumes all psr have same sigma p. This will need updating
    
    exp_term = 1 - np.exp(-γ*dt) #precompute exponential terms since it is used often. Note if gamma isnt changing we could move this elsewhere. TODO
    σp2 = σp**2

    term_11 = σp2 * (γ*dt - exp_term - 0.5*exp_term**2) / γ**3
    term_12 = σp2 * (exp_term**2) / (2*γ**2)
    term_21 = term_12 
    term_22 = σp2* (1 - np.exp(-2*γ*dt)) / (2*γ)


    component_array = np.array([[term_11,term_12],
                                [term_21,term_22]]) 

    return block_diag_view_jit(component_array,Npsr)  
    
"""
The R matrix as a scalar - same noise covariance for all pulsars
"""
@njit(fastmath=True)
def R_function(sigma_m,Npsr):
    return np.eye(Npsr)*sigma_m**2
    
