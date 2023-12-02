import numpy as np 
from numba import njit
from model import F_function,R_function,Q_function # H function is defined via a class init. #todo change this! we want things defined in a class that is read by KalmanFilter

import sys



"""
Kalman update step for diagonal matrices where everything is considered as a 1d vector
"""
#@njit(fastmath=True)
def update(x, P, observation,R,GW,ephemeris):

    #Construct the H matrix. Might be smarter way to do this which avoids the loop. TODO
    N = int(len(x)/2)
    H = np.zeros((N, 2*N))
    for i in range(N):
        H[i, 2*i] = 1
        H[i, 2*i + 1] = GW[i]


    y_predicted = H@x - GW*ephemeris

    y    = observation - y_predicted
    S    = H@P@H.T + R 
    Sinv = np.linalg.inv(S)
    K    = P@H.T@Sinv 
    xnew = x + K@y
   

    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    # P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    # and works for non-optimal K vs the equation
    # P = (I-KH)P usually seen in the literature.
    I_KH = 1.0 - K@H
    Pnew = I_KH @ P @ I_KH.T + K @ R @ K.T

 
    #Map back from the state to measurement space. We can surface and plot this variable
    ypred = H@xnew - GW*ephemeris

    return xnew, Pnew,likelihood_value,ypred


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
@njit(fastmath=True)
def predict(x,P,F,Q): 
    xp = F@x
    Pp = F@P@F.T + Q  


    return xp,Pp



"""
Given a Bilby dict, make it a numpy array 
"""
def dict_to_array(some_dict,target_keys):
    selected_dict = {k:some_dict[k] for k in target_keys}
    return np.array(list(selected_dict.values())).flatten()




class KalmanFilter:
    """
    A class to implement the Kalman filter.

    It takes two initialisation arguments:

        `Model`: definition of all the Kalman machinery: state transition models, covariance matrices etc. 

        `Observations`: class which holds the noisy observations recorded at the detector

    """

    def __init__(self,Model, Observations,PTA):

        """
        Initialize the class. 
        """

        self.model = Model
        self.observations = Observations

        #PTA related quantities
        self.dt = PTA.dt
        self.q = PTA.q
        self.t = PTA.t
        self.q = PTA.q
        self.q_products = PTA.q_products
        self.ephemeris = PTA.ephemeris

        self.Npsr   = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]
        
        self.H_function = Model.H_function




        # Define a list_of_keys arrays. 
        # This is useful for parsing the Bibly dictionary into arrays efficiently
        # There may be a less verbose way of doing this, but works well in practice
        self.list_of_f_keys        = [f'f0{i}' for i in range(self.Npsr)]
        self.list_of_fdot_keys     = [f'fdot{i}' for i in range(self.Npsr)]
        self.list_of_gamma_keys    = [f'gamma{i}' for i in range(self.Npsr)]
        self.list_of_distance_keys = [f'distance{i}' for i in range(self.Npsr)]
        self.list_of_sigma_p_keys  = [f'sigma_p{i}' for i in range(self.Npsr)]
        





    def parse_dictionary(self,parameters_dict):
        
        
        #All the GW parameters can just be directly accessed as variables
        omega_gw = parameters_dict["omega_gw"]
        phi0_gw  = parameters_dict["phi0_gw"]
        psi_gw   = parameters_dict["psi_gw"]
        iota_gw  = parameters_dict["iota_gw"]
        delta_gw = parameters_dict["delta_gw"]
        alpha_gw = parameters_dict["alpha_gw"]
        h        = parameters_dict["h"]

        #Now read in the pulsar parameters. Explicit.
        f       = dict_to_array(parameters_dict,self.list_of_f_keys)
        fdot    = dict_to_array(parameters_dict,self.list_of_fdot_keys)
        d       = dict_to_array(parameters_dict,self.list_of_distance_keys)
  
    
        gamma = parameters_dict["gamma"]
        sigma_p = parameters_dict["sigma_p"]

        #Other noise parameters
        sigma_m = parameters_dict["sigma_m"]

        return omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
               f,fdot,gamma,d,sigma_p,\
               sigma_m



    def likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,gamma,d,sigma_p,\
        sigma_m = self.parse_dictionary(parameters) #todo I think this is our bottleneck
        
        #Precompute transition/Q/R Kalman matrices
        #F,Q,R are time-independent functions of the parameters
        F = F_function(gamma,self.dt,self.Npsr)
        R = R_function(sigma_m,self.Npsr)
        Q = Q_function(gamma,sigma_p,self.dt,self.Npsr)
     
        #Initialise x and P
        x=  np.zeros(2*self.Npsr) #guess of intial states. Assume for every pulsar the heterodyned phase/frequency = 0
        P=  np.eye(2*self.Npsr)* sigma_m * 1e10 #Guess that the uncertainty in the initial state is a few orders of magnitude greater than the measurement noise
        #Does it make sense for this to be diagonal? Probably not...?




        # Precompute the influence of the GW
        # This is solely a function of the parameters and the t-variable but NOT the states
        GW = self.H_function(delta_gw,
                                   alpha_gw,
                                   psi_gw,
                                   self.q,
                                   self.q_products,
                                   h,
                                   iota_gw,
                                   omega_gw,
                                   self.t,
                                   phi0_gw,
                                   d
                                )
        
        
        #Initialise the likelihood
        likelihood = 0.0
              
       
        #Do the first update step
        x,P,likelihood_value,y_predicted = update(x,P, self.observations[0,:],R,GW[0,:],self.ephemeris[0,:])
        likelihood +=likelihood_value




        #Place to store results
        x_results = np.zeros((self.Nsteps,2*self.Npsr))
        y_results = np.zeros((self.Nsteps,self.Npsr))

        x_results[0,:] = x
        y_results[0,:] = y_predicted
       


        for i in np.arange(1,self.Nsteps):

            obs                              = self.observations[i,:]                                     #The observation at this timestep
            x_predict, P_predict             = predict(x,P,F,Q)                                           #The predict step
            x,P,likelihood_value,y_predicted = update(x,P, self.observations[i,:],R,GW[i,:],self.ephemeris[i,:]) #The update step    
            
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       
   
        return likelihood,x_results,y_results



        
        






