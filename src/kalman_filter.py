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


@njit(fastmath=True)
def construct_H_matrix(some_vector):
   
    n = len(some_vector)
    
    some_matrix = np.zeros(2 * n**2)
    step = 2 * (n + 1)
    some_matrix[::step] = 1
    some_matrix[1::step] = some_vector
    some_matrix = some_matrix.reshape(n, 2*n)

    return some_matrix



"""
Kalman update step
"""
#@njit(fastmath=True)
def update(phi,f,x, P,Ptest,observation,R,GW,ephemeris):
    

    
    A,B,C,D = np.split(Ptest,4)
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    D = D.flatten()

    
    #print()
    y_predicted = phi - f*GW - GW*ephemeris
    y           = observation - y_predicted #The residual w.r.t actual data
    S           = A + GW * B + GW * C + (GW**2) * D + R
    
    print("Covar S = ", S, A)

    # Calculate odd elements (indexing starts from 0)
    K = np.zeros(2 * len(S))

    Kodd = (A + GW * B)[::2] / S[::2]
    Keven = (C + GW * D)[1::2] / S[1::2]
    K[::2] = Kodd

    # Calculate even elements
    K[1::2] = Keven


    xnew = x + K*np.repeat(y,2)


    Anew = A*((1-Kodd) + C*(1-Kodd*GW))
    Bnew = B*((1-Kodd) + D*(1-Kodd*GW))
    Cnew = A*((1-Keven) + C*(1-Keven*GW))
    Dnew = B*((1-Keven) + D*(1-Keven*GW))
    
    Pnew = np.array([Anew,Bnew,Cnew,Dnew])
    ll          = log_likelihood(y,S)       #and get the likelihood
    print("likelihood=", ll)

    if np.isnan(ll):
        sys.exit()
    


    #H           = construct_H_matrix(GW)    #Determine the H matrix for this step
    #y_predicted1 = H@x - GW*ephemeris        #The predicted y

    #print(y_predicted)
    #print(y_predicted1)
    
    
   
    #HP          = H@P                       #Precompute H@P as a variable since we use it twice. Probably offers no real performance gain...
    #S1           = np.diag(HP@H.T) + R       #S is diagonal
    #S1           = HP@H.T + np.eye(2)*R       #S is diagonal
    #print(S1)


    # print('S')
    # print(S)
    # print('S1')
    # print(S1)


    #print(S1.shape)
    #print(S)
    #print(S1)   
    #sinv = np.linalg.inv(S1)
    #K = P@H.T@sinv        #making use of diagonality of S


 

    #print(S1)
    #sys.exit()
    
    #y_predicted = H@x - GW*ephemeris        #The predicted y
    

    #K           = np.divide(P@H.T,S)        #making use of diagonality of S
    #xnew        = x + K@y                   #update x


 

    #print(xnew)
    #print(xnew1)
    #sys.exit()


    
    #Pnew        =  P - K@HP                 #update P, using earlier defined HP
   
    
    return xnew, Pnew,ll
    
"""
Kalman predict step
"""
@njit(fastmath=True)
def predict(x,P,F,F_transpose,Q): 


    
    A,B,C,D = np.split(P,4)
    A = A.flatten()
    B = B.flatten()
    C = C.flatten()
    D = D.flatten()


    #split x into phi and f 
    state_phi = x[0::2]
    state_f = x[1::2]


    #Read in F
    Fa = F[0,0]
    Fb = F[0,1]
    Fc = F[1,1]

    #State predicitns
    phi_predict = Fa*state_phi+Fb*state_f
    f_predict = Fc*state_f


    #Covar predictions
    Qa = Q[0,0]
    Qb = Q[0,1]
    Qc = Q[1,0]
    Qd = Q[1,1]

    Ap = Fa*(A*Fa+B*Fb) + Fb*(C*Fa + D*Fb) + Qa
    Bp = Fa*B*Fc + Fb*D*Fc + Qb
    Cp = Fc*(C*Fa + D*Fb) + Qc
    Dp = D*Fc**2 + Qd

    print('Input A = ', A)
    print("predicted A = ", Ap)

    #xp = np.array([])

    #return F@x,F@P@F_transpose + Q  
    return phi_predict, f_predict, Ap,Bp,Cp,Dp


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
        #self.list_of_distance_keys = [f'distance{i}' for i in range(self.Npsr)]
        self.list_of_chi_keys = [f'chi{i}' for i in range(self.Npsr)]

        self.list_of_sigma_p_keys  = [f'sigma_p{i}' for i in range(self.Npsr)]
        

        #some initiaisation
        self.x0=  np.zeros(2*self.Npsr) #guess of intial states. Assume for every pulsar the heterodyned phase/frequency = 0
        
        
        #How to initialise P ?
        #Option A - canonical
        #self.P0=  np.eye(2*self.Npsr)* sigma_m * 1e5 #Guess that the uncertainty in the initial state is a few orders of magnitude greater than the measurement noise

        #Option B - zeroes
        self.P0=  np.eye(2*self.Npsr)* 0.0

        #Option C - different for phi and f
        component_array = np.array([[0.0,0.0],
                                    [0.0,1e-3]])

        #self.P0 = block_diag_view_jit(component_array,self.Npsr) #we need to apply this over N pulsars

        #self.R = R_function(PTA.σm,self.Npsr)
        self.R = PTA.σm**2
        self.F,self.Q = Model.kalman_machinery()

        #We can also precompute the transpose of F
        self.F_transpose = self.F.T


        #Initialise Kalman machinery







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

        #Other noise parameters
        #sigma_m = parameters_dict["sigma_m"]#.item() #float, known

        return omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
               f,fdot,chi,sigma_p#,\
              # sigma_m





    def likelihood(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,chi,sigma_p = self.parse_dictionary(parameters) 
        
        #Precompute transition/Q/R Kalman matrices
        #F,Q,R are time-independent functions of the parameters
        #F = F_function(gamma,self.dt,self.Npsr)
        #R = R_function(sigma_m,self.Npsr)
        #Q = Q_function(gamma,sigma_p,self.dt,self.Npsr)
        F = self.F 
        F_transpose = self.F_transpose
        Q = self.Q * sigma_p**2
     
        #Initialise x and P
        x = self.x0 
        P = self.P0

        P[0,0] = 1e-5
        P[0,1] = 1e-5
        P[1,0] = 1e-5
        P[1,1] = 1e-5


        P[2,2] = 1e-5
        P[2,3] = 1e-5
        P[3,2] = 1e-5
        P[3,3] = 1e-5


        Ptest = np.ones(8)*1e-5


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

        #Define an ephemeris correction
        ephemeris = f + np.outer(self.t,fdot) #ephemeris correction
        
        
        #Initialise the likelihood
        ll = 0.0
              
       

        #split x into phi and f 
        state_phi = x[0::2]
        state_f = x[1::2]

        #Do the first update step
        x,P,likelihood_value = update(state_phi,state_f,x,P,Ptest, self.observations[0,:],self.R,GW[0,:],ephemeris[0,:])
        ll +=likelihood_value

        
        #y_results[0,:] = y_predicted 
        for i in np.arange(1,self.Nsteps):
            obs                              = self.observations[i,:]                                     #The observation at this timestep
            #x_predict, P_predict             = predict(x,P,F,F_transpose,Q)   
            
            phi_predict, f_predict, Ap,Bp,Cp,Dp = predict(x,P,F,F_transpose,Q)  
            Ptest = np.array([Ap,Bp,Cp,Dp])
                                                    #The predict step
            x,P,likelihood_value = update(phi_predict,f_predict,x,P,Ptest, self.observations[i,:],self.R,GW[i,:],ephemeris[i,:]) #The update step    
            ll +=likelihood_value

         
        return ll




    def likelihood_plotter(self,parameters):

        #Map from the dictionary into variables and arrays
        omega_gw,phi0_gw,psi_gw,iota_gw,delta_gw,alpha_gw,h,\
        f,fdot,chi,sigma_p = self.parse_dictionary(parameters) 
        
        #Precompute transition/Q/R Kalman matrices
        F = self.F 
        F_transpose = self.F_transpose
        Q = self.Q * sigma_p**2
     
        #Initialise x and P
        x = self.x0 
        P = self.P0

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


        #Define an ephemeris correction
        ephemeris = f + np.outer(self.t,fdot) #ephemeris correction
        
        
        #Initialise the likelihood
        ll = 0.0
              
       
        #Do the first update step
        x,P,likelihood_value = update(x,P, self.observations[0,:],self.R,GW[0,:],ephemeris[0,:])
        ll +=likelihood_value




        #Place to store results
        x_results = np.zeros((self.Nsteps,2*self.Npsr))
        y_results = np.zeros((self.Nsteps,self.Npsr))

        
        #y_results[0,:] = y_predicted 
        for i in np.arange(1,self.Nsteps):
            obs                              = self.observations[i,:]                                     #The observation at this timestep
            x_predict, P_predict             = predict(x,P,F,F_transpose,Q)                                           #The predict step
            x,P,likelihood_value = update(x_predict,P_predict, self.observations[i,:],self.R,GW[i,:],ephemeris[i,:]) #The update step    
            ll +=likelihood_value

            H              = construct_H_matrix(GW[i,:])    #Determine the H matrix for this step
            y_predicted =  H@x - GW[i,:]*ephemeris[i,:]
    
            x_results[i,:] = x
            y_results[i,:] = y_predicted
       
     
        return ll,x_results,y_results



        
        
        
        






