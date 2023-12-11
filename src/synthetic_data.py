

import sdeint
# import numpy as np 

from gravitational_waves import gw_psr_terms


import logging
import numpy as np 
from utils import block_diag_view_jit
import numpy as np 

class SyntheticData:
    
    

    def __init__(self,pulsars,P):


        #Discrete timesteps
        self.t = pulsars.t


        γ = pulsars.γ[0] #TODO gamma should just be a scalar
        σp= pulsars.σp[0] #todo this should NOT be a scalar, but a different value for every pulsar. Need to construct array differently 

        
        #Integrate the 2D vector Ito equation dx = Ax dt + BdW
        #We assume the state is x = (phi,f).
        # For e.g. 2 pulsars it is x=(phi_1,f_1,phi_2,f_2) 
        component_array_a = np.array([[0,1],
                                    [0,-γ]]) #See pg2 of O'Leary in docs


        component_array_b = np.array([[0,0],
                                    [0,σp]]) #See pg2 of O'Leary in docs


        A = block_diag_view_jit(component_array_a,pulsars.Npsr) #we need to apply this over N pulsars
        B = block_diag_view_jit(component_array_b,pulsars.Npsr) #we need to apply this over N pulsars
        x0 = np.zeros((2*pulsars.Npsr)) # Initial condition. All initial phases are zero and heterodyned frequencies 


        #Random seeding
        generator = np.random.default_rng(P.seed)


        #Integrate the state equation
        #e.g. https://pypi.org/project/sdeint/
        def f(x,t):
            return A.dot(x)
        def g(x,t):
            return B

        state= sdeint.itoint(f,g,x0, self.t,generator=generator) #This has shape (Ntimes x 2*Npsr)
        
        #It is useful to have phi/f separatly:
        self.state_phi = state[:,0::2]
        self.state_f = state[:,1::2]


        # #Now calculate the modulation factor due to the GW   
        #GW_function = gw_psr_terms 
        # 
        #      
        # if P.use_psr_terms_in_data:
            
        #     logging.info("You are including the PSR terms in your synthetic data generation")
        # else:
        #     GW_function = gw_earth_terms
        #     logging.info("You are using just the Earth terms in your synthetic data generation")




        #Now get the heterodyned phi measured at the detector
        GW = gw_psr_terms(
                                        P.δ,
                                        P.α,
                                        P.ψ,
                                        pulsars.q,
                                        pulsars.q_products,
                                        P.h,
                                        P.ι,
                                        P.Ω,
                                        pulsars.t,
                                        P.Φ0,
                                        pulsars.χ
                                        )
 
       
        self.phi_measured_no_noise = self.state_phi - (self.state_f+pulsars.ephemeris)*GW
        measurement_noise = generator.normal(0, pulsars.σm,self.phi_measured_no_noise.shape) # Measurement noise. Seeded
        self.phi_measured = self.phi_measured_no_noise + measurement_noise



