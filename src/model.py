

import numpy as np
from numba import njit

import logging 
from gravitational_waves import gw_psr_terms,gw_earth_terms,null_model
import sys
from utils import block_diag_view_jit


class PhaseModel:

    """
    A linear model of the state evolution x = (phi,f)
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

            #F-matrix
            component_array = np.array([[1,(1-np.exp(-γ*dt))/γ],
                                        [0,np.exp(-γ*dt)]]) 

            F = block_diag_view_jit(component_array,Npsr) 

            #Q-matrix
            exp_term = 1 - np.exp(-γ*dt) #precompute exponential terms since it is used often. Note if gamma isnt changing we could move this elsewhere. TODO

            term_11 = (γ*dt - exp_term - 0.5*exp_term**2) / γ**3
            term_12 = (exp_term**2) / (2*γ**2)
            term_21 = term_12 
            term_22 = (1 - np.exp(-2*γ*dt)) / (2*γ)


            component_array = np.array([[term_11,term_12],
                                        [term_21,term_22]]) 
            Q = block_diag_view_jit(component_array,Npsr)  

            return F, Q

    def optimised_kalman_machinery(self):

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


