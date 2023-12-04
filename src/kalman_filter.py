import numpy as np 
from numba import njit
from model import F_function,R_function,Q_function # H function is defined via a class init. #todo change this! we want things defined in a class that is read by KalmanFilter

from scipy.stats import multivariate_normal
import sys

from utils import block_diag_view_jit

from scipy.sparse import csr_array,csc_array,identity
from scipy.sparse.linalg import inv as sparse_inv


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
    log_det_cov = np.sum(np.log(np.diag(cov))) # Uses log rules and diagonality of covariance matrix
    ll = -0.5 * (log_det_cov + y@x+ N*np.log(2*np.pi))
    return ll


"""
Kalman update step
"""
@njit(fastmath=True)
def update(x, P, observation,R,GW,ephemeris):

    #Construct the H matrix. Might be smarter way to do this which avoids the loop. TODO
    N = int(len(x)/2)
    H = np.zeros((N, 2*N))

    for i in range(N):
        H[i, 2*i] = 1
        H[i, 2*i + 1] = -GW[i]


    y_predicted = H@x - GW*ephemeris
    y           = observation - y_predicted
    S           = H@P@H.T + R
    Sinv        = np.linalg.inv(S) #S is diagonal so we can compute the inverse differently here
    K           = P@H.T@Sinv
    xnew        = x + K@y


    #Update the covariance 
    #Following FilterPy https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/EKF.py by using
    #P = (I-KH)P(I-KH)' + KRK' which is more numerically stable
    #and works for non-optimal K vs the equation
    #P = (I-KH)P usually seen in the literature.
    I_KH = np.eye(2*N) - K@H
    Pnew = I_KH @ P @ I_KH.T + K @ R @ K.T
    
    ll = log_likelihood(y,np.diag(S))

    #Map back from the state to measurement space. We can surface and plot this variable
    #ypred = H@xnew - GW*ephemeris
    return xnew, Pnew,ll #,0.0
    #return x, P,0.0,0.0


"""
Kalman predict step for diagonal matrices where everything is considered as a 1d vector
"""
@njit(fastmath=True)
def predict(x,P,F,Q): 
    #xp = F@x
    #Pp = F@P@F.T + Q  
    return F@x,F@P@F.T + Q 



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
        #self.ephemeris = PTA.ephemeris

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
        

        #some initiaisation
        self.x0 = np.zeros(2*self.Npsr) #guess of intial states. Assume for every pulsar the heterodyned phase/frequency = 0

        self.P0 = np.eye(2*self.Npsr)* 0.0

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
        d       = dict_to_array(parameters_dict,self.list_of_distance_keys)
  
        #TODO. For now gamma and sigma_p are shared between all pulsars
        gamma = parameters_dict["gamma"].item()
        sigma_p = parameters_dict["sigma_p"].item()

        #Other noise parameters
        sigma_m = parameters_dict["sigma_m"]#.item() #float, known

        return omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
               f,fdot,gamma,d,sigma_p,\
               sigma_m




    def fast_likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,gamma,d,sigma_p,\
        sigma_m = self.parse_dictionary(parameters) 
        
        #Precompute transition/Q/R Kalman matrices
        #F,Q,R are time-independent functions of the parameters
        F = F_function(gamma,self.dt,self.Npsr)
        R = R_function(sigma_m,self.Npsr)
        Q = Q_function(gamma,sigma_p,self.dt,self.Npsr)
     
        #Initialise x and P
        x= self.x0 
        P= self.P0

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


        # Define an ephemeris correction
        ephemeris = f + np.outer(self.t,fdot) #ephemeris correction
        
        
        #Initialise the likelihood
        ll = 0.0
              
       
        # Do the first update step
        x,P,likelihood_value,y_predicted = update(x,P, self.observations[0,:],R,GW[0,:],ephemeris[0,:])
        #ll +=likelihood_value
        

        for i in np.arange(1,self.Nsteps):
            obs                              = self.observations[i,:]                                     #The observation at this timestep
            x_predict, P_predict             = predict(x,P,F,Q)                                           #The predict step
            #x,P,likelihood_value,y_predicted = update(x_predict,P_predict, self.observations[i,:],R,GW[i,:],ephemeris[i,:]) #The update step    
            x,P,likelihood_value,y_predicted = update(x,P, self.observations[i,:],R,GW[i,:],ephemeris[i,:]) #The update step    

            #ll +=likelihood_value
        
        
        
        return 0.0
















    def likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,gamma,d,sigma_p,\
        sigma_m = self.parse_dictionary(parameters) 
        
        #Precompute transition/Q/R Kalman matrices
        #F,Q,R are time-independent functions of the parameters
        F = F_function(gamma,self.dt,self.Npsr)
        R = R_function(sigma_m,self.Npsr)
        Q = Q_function(gamma,sigma_p,self.dt,self.Npsr)
     
        #Initialise x and P
        x=  np.zeros(2*self.Npsr) #guess of intial states. Assume for every pulsar the heterodyned phase/frequency = 0
        
        
        

        #How to initialise P ?
        #Option A - canonical
        #P=  np.eye(2*self.Npsr)* sigma_m * 1e5 #Guess that the uncertainty in the initial state is a few orders of magnitude greater than the measurement noise

        #Option B - zeroes
        P=  np.eye(2*self.Npsr)* 0.0

        #Option C - different for phi and f
        component_array = np.array([[0.0,0.0],
                                    [0.0,1e-3]])

        #P = block_diag_view_jit(component_array,self.Npsr) #we need to apply this over N pulsars
  
        

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


        #Define an ephemeris correction
        ephemeris = f + np.outer(self.t,fdot) #ephemeris correction
        
        
        #Initialise the likelihood
        ll = 0.0
              
       
        #Do the first update step
        x,P,likelihood_value,y_predicted = update(x,P, self.observations[0,:],R,GW[0,:],ephemeris[0,:])
        ll +=likelihood_value




        #Place to store results
        x_results = np.zeros((self.Nsteps,2*self.Npsr))
        y_results = np.zeros((self.Nsteps,self.Npsr))

        #P_results = np.zeros((self.Nsteps,self.Npsr))

        
        y_results[0,:] = y_predicted 
        for i in np.arange(1,self.Nsteps):
            obs                              = self.observations[i,:]                                     #The observation at this timestep
            x_predict, P_predict             = predict(x,P,F,Q)                                           #The predict step
            x,P,likelihood_value,y_predicted = update(x_predict,P_predict, self.observations[i,:],R,GW[i,:],ephemeris[i,:]) #The update step    
            ll +=likelihood_value
            #print(P)
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       
     
        return ll,x_results,y_results



        
        






