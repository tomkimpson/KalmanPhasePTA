

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

    def __init__(self,P,PTA):

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


        self.γ = PTA.γ[0] #assume gamma is known and it is the same for every pulsar
        self.dt = PTA.dt
        self.Npsr = PTA.Npsr



    def kalman_machinery(self):

            γ    = self.γ 
            dt   = self.dt 
            Npsr = self.Npsr 



            Fx = 1.0
            Fy = (1-np.exp(-γ*dt))/γ
            Fz = np.exp(-γ*dt)

        

            #Q-matrix
            exp_term = 1 - np.exp(-γ*dt) #precompute exponential terms since it is used often. Note if gamma isnt changing we could move this elsewhere. TODO

            Qa = (γ*dt - exp_term - 0.5*exp_term**2) / γ**3
            Qb = (exp_term**2) / (2*γ**2)
            Qc = Qb
            Qd = (1 - np.exp(-2*γ*dt)) / (2*γ)




            return Fx,Fy,Fz, Qa,Qb,Qc,Qd


