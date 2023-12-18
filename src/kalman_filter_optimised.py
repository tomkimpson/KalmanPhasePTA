import numpy as np 
from numba import njit
#from model import R_function # H function is defined via a class init. #todo change this! we want things defined in a class that is read by KalmanFilter


import sys




"""
Kalman likelihood
"""
@njit(fastmath=True)
def log_likelihood(y,cov):
    N = len(y)
    x = y/cov
    #The innovation covariance is diagonal
    #Therefore we can calculate the determinant of its logarithm as below
    #A normal np.linalg.det(cov)fails due to overflow, since we just have
    #det ~ 10^{-13} * 10^{-13} * 10^{-13}*...
    log_det_cov = np.sum(np.log(cov)) # Uses log rules and diagonality of covariance matrix
    ll = -0.5 * (log_det_cov + y@x+ N*np.log(2*np.pi))
    return ll


"""
Kalman update step
"""
@njit(fastmath=True)
def update(xφ,xf,A,B,C,D,observation,R,GW,ephemeris):

    #First lets get the likelihood
    y_predicted = xφ - xf*GW - GW*ephemeris #The predicted y
    y           = observation - y_predicted #The residual w.r.t actual data


    #Precompute useful repeated quantities
    #I am not sure this offers any significant speed up
    GW_b = GW*B 
    GW_c = GW*C


    S           = A - GW_b - GW_c + (GW**2)*D + R  
    ll          = log_likelihood(y,S)       #and get the likelihood 

    #Now lets update x and P

    Kodd = (A - GW_b)/S
    Keven = (C - GW_c)/S

    #Update state estimates
    #xφ_new = xφ + Kodd*y
    #xf_new = xf + Keven*y

    #Update covariances
    K1 = 1.0-Kodd
    K2 = 1.0+Keven*GW
    A_new = A*K1 + Kodd*GW_c
    B_new = B*K1 + Kodd*D*GW
    C_new = -A*Keven + C*K2
    D_new = -B*Keven + D*K2

    return xφ + Kodd*y,xf + Keven*y, A_new,B_new,C_new,D_new,ll
    
"""
Kalman predict step
"""
@njit(fastmath=True)
def predict(xφ,xf,A,B,C,D,Fx,Fy,Fz,Qa,Qb,Qc,Qd): 

    xφ_predict = Fx*xφ + Fy*xf
    xf_predict = Fz*xf

    A_predict = Fx*(A*Fx + C*Fy) + Fy*(B*Fx +D*Fy)
    B_predict = (B*Fx + D*Fy)*Fz 
    C_predict = C*Fx*Fz + D*Fy*Fz 
    D_predict =  D*Fz**2 

    return xφ_predict,xf_predict,A_predict+Qa,B_predict+Qb,C_predict+Qc,D_predict+Qd


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

        self.Npsr   = self.observations.shape[-1]
        self.Nsteps = self.observations.shape[0]
        


        # Define a list_of_keys arrays. 
        # This is useful for parsing the Bibly dictionary into arrays efficiently
        # There may be a less verbose way of doing this, but works well in practice
        self.list_of_f_keys        = [f'f0{i}' for i in range(self.Npsr)]
        self.list_of_fdot_keys     = [f'fdot{i}' for i in range(self.Npsr)]
        self.list_of_gamma_keys    = [f'gamma{i}' for i in range(self.Npsr)]
        self.list_of_chi_keys      = [f'chi{i}' for i in range(self.Npsr)]
        self.list_of_sigma_p_keys  = [f'sigma_p{i}' for i in range(self.Npsr)]
        
        
        #Define some Kalman matrices
        self.R = PTA.σm**2
        self.Fx,self.Fy,self.Fz,self.Qa,self.Qb, self.Qc, self.Qd = Model.optimised_kalman_machinery()
        self.H_function = Model.H_function

        self.x0 =  np.zeros((self.Npsr))





    """
    Bilby provides samples from prior as a dict
    Read these into variables.
    For the pulsar parameters, read these into vectors where the n-th element corresponds to the n-th pulsar 
    """
    def parse_dictionary(self,parameters_dict):
        
        
        #All the GW parameters can just be directly accessed as variables
        omega_gw = parameters_dict["omega_gw"].item()
        phi0_gw  = parameters_dict["phi0_gw"].item()
        psi_gw   = parameters_dict["psi_gw"].item()
        iota_gw  = parameters_dict["iota_gw"].item()
        delta_gw = parameters_dict["delta_gw"].item()
        alpha_gw = parameters_dict["alpha_gw"].item()
        h        = parameters_dict["h"].item()

        #Now read in the pulsar parameters. Explicit.
        f       = dict_to_array(parameters_dict,self.list_of_f_keys)
        fdot    = dict_to_array(parameters_dict,self.list_of_fdot_keys)
        chi     = dict_to_array(parameters_dict,self.list_of_chi_keys)
  
        #TODO. For now gamma and sigma_p are shared between all pulsars
        #gamma = parameters_dict["gamma"].item()
        sigma_p = parameters_dict["sigma_p"].item()


        return omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
               f,fdot,chi,sigma_p





    def likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,chi,sigma_p = self.parse_dictionary(parameters) 
      
        
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
                                   chi
                                )

        Qa = self.Qa * sigma_p**2
        Qb = self.Qb * sigma_p**2
        Qc = self.Qc * sigma_p**2
        Qd = self.Qd * sigma_p**2

        #Define an ephemeris correction
        ephemeris = f + np.outer(self.t,fdot) #ephemeris correction
        
        
        #Initialise the likelihood
        ll = 0.0
              

        #Split the state x into phi and f
        xφ = self.x0
        xf = self.x0

        #...and the covariance
        A = self.x0
        B = self.x0
        C = self.x0
        D = self.x0


       
        #Do the first update step
        xφ,xf, A,B,C,D,likelihood_value = update(xφ,xf,A,B,C,D,self.observations[0,:],self.R,GW[0,:],ephemeris[0,:])
        ll +=likelihood_value
  
        for i in np.arange(1,self.Nsteps):
            xφ_predict, xf_predict, A_predict, B_predict, C_predict, D_predict = predict (xφ,xf,A,B,C,D,self.Fx,self.Fy,self.Fz,Qa,Qb,Qc,Qd)                                   #The observation at this timestep
            xφ,xf, A,B,C,D,likelihood_value = update(xφ_predict, xf_predict, A_predict, B_predict, C_predict, D_predict, self.observations[i,:],self.R,GW[i,:],ephemeris[i,:]) #The update step    
            ll +=likelihood_value

         
        return ll







