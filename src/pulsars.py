
from numpy import sin, cos
import numpy as np 
import pandas as pd 
import logging
#from utils import get_project_root
#from gravitational_waves import principal_axes

from pathlib import Path
import os 


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Pulsars:


    def __init__(self,SystemParameters):



        #Define some universal constants
        pc = 3e16     # parsec in m
        c  = 3e8      # speed of light in m/s


        #Load the pulsar data
        root = get_project_root()
        pulsars = pd.read_csv(root / "data/NANOGrav_pulsars.csv")

        if SystemParameters.Npsr != 0:
            pulsars = pulsars.sample(SystemParameters.Npsr,random_state=SystemParameters.seed) #can also use  pulsars.head(N) to sample  

        
        #Extract the parameters
        self.f         = pulsars["F0"].to_numpy()            # Hz
        self.fdot      = pulsars["F1"] .to_numpy()           # s^-2
        self.d         = pulsars["DIST"].to_numpy()*1e3*pc/c # this is in units of s^-1
        self.γ         = np.ones_like(self.f) * 1e-13       # for every pulsar let γ be 1e-13. Hardcoded value, i.e. not a variable in SystemParameters
        self.δ         = pulsars["DECJD"].to_numpy()                 # radians
        self.α         = pulsars["RAJD"].to_numpy()                  # radians

    
        #Pulsar positions as unit vectors
        self.q         = _unit_vector(np.pi/2.0 -self.δ, self.α) # 3 rows, N columns


        #Precompute the q^i q^j terms
        self.q_products=_precomute_q_terms(self.q)



        #Discrete timesteps
        self.dt      = SystemParameters.cadence * 24*3600 #from days to step_seconds
        end_seconds  = SystemParameters.T* 365*24*3600 #from years to second
        self.t       = np.arange(0,end_seconds,self.dt)
        

        #Process noise 
        if SystemParameters.process_noise is True:
            #self.σp = Read the true value from file. TODO
            pass

        elif SystemParameters.process_noise == 'Fixed':
            self.σp = np.ones_like(self.f)*SystemParameters.σp

        elif SystemParameters.process_noise == 'Random':
            generator = np.random.default_rng(SystemParameters.seed)
            self.σp   = generator.uniform(low = SystemParameters.σp/10,high=SystemParameters.σp*10,size=self.Npsr)

        #     logging.info("You are assigning the σp terms randomly")
        # else:
        #     self.σp = np.full(self.Npsr,SystemParameters.σp)








        # #Assign some other useful quantities to self
        # #Some of these are already defined in SystemParameters, but I don't want to pass
        # #the SystemParameters class to the Kalman filter - it should be completely blind
        # #to the true parameters - it only knows what we tell it!
        self.Npsr    = len(self.f) 
        self.σm =  SystemParameters.σm
        self.ephemeris = self.f + np.outer(self.t,self.fdot) 
     
        
        
        


"""
Given a latitude theta and a longitude phi, get the xyz unit vector which points in that direction 
"""
def _unit_vector(theta,phi):
    qx = sin(theta) * cos(phi)
    qy = sin(theta) * sin(phi)
    qz = cos(theta)
    return np.array([qx, qy, qz]).T


"""Precompute the 9 cross terms i.e. xx,xy,xz,yx,yy,yz,zx,zy,zz.
"""
def _precomute_q_terms(q):

    Npsr = q.shape[0]
    q_products = np.zeros((Npsr,9))
    k = 0
    for n in range(Npsr):
        k = 0
        for i in range(3):
            for j in range(3):
                q_products[n,k] = q[n,i]*q[n,j]
                k+=1

    return q_products.T #transpose is used so shape is (9,Npsr)