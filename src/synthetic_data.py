

import sdeint
# import numpy as np 

# from gravitational_waves import gw_earth_terms,gw_psr_terms


import logging
import numpy as np 
import numba as nb

@nb.jit(nopython=True) #From https://stackoverflow.com/a/58017351
def block_diag_view_jit(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num * rows, num * cols), dtype=arr.dtype)
    for k in range(num):
        result[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols] = arr
    return result








class SyntheticData:
    
    

    def __init__(self,pulsars,P):


        #Discrete timesteps
        t = pulsars.t



        γ = pulsars.γ[0] #TODO gamma shoud just be a scalar
        σp= pulsars.σp[0] #todo this should NOT be a scalar, but a different value for every pulsar. Need to construct array differently 

        

        #Integrate the 2D vector Ito equation dx = Ax dt + BdW
        component_array_a = np.array([[0,1],
                                    [0,-γ]]) #See pg2 of O'Leary in docs


        component_array_b = np.array([[0,0],
                                    [0,σp]]) #See pg2 of O'Leary in docs


        A = block_diag_view_jit(component_array_a,pulsars.Npsr) #we need to apply this over N pulsars
        B = block_diag_view_jit(component_array_b,pulsars.Npsr) #we need to apply this over N pulsars
        x0 = np.zeros((2*pulsars.Npsr)) #initial condition. All initial phases are zero and heterodyned frequencies 





        #Random seeding
        generator = np.random.default_rng(P.seed)


        #Integrate the state equation
        #e.g. https://pypi.org/project/sdeint/
        def f(x,t):
            return A.dot(x)
        def g(x,t):
            return B

        self.state= sdeint.itoint(f,g,x0, t,generator=generator)


        print(self.state)
        
        # #Turn σp and γ into diagonal matrices that can be accepted by vectorized sdeint
        # σp = np.diag(σp)
        # γ = np.diag(γ)



        # 

        # #Now calculate the modulation factor due to the GW
        
        # if P.use_psr_terms_in_data:
        #     GW_function = gw_psr_terms
        #     logging.info("You are including the PSR terms in your synthetic data generation")
        # else:
        #     GW_function = gw_earth_terms
        #     logging.info("You are using just the Earth terms in your synthetic data generation")

        # X_factor = GW_function(
        #                                 P.δ,
        #                                 P.α,
        #                                 P.ψ,
        #                                 pulsars.q,
        #                                 pulsars.q_products,
        #                                 P.h,
        #                                 P.ι,
        #                                 P.Ω,
        #                                 pulsars.t,
        #                                 P.Φ0,
        #                                 pulsars.chi
        #                                 )
            
        
        # #The measured frequency, no noise
        # self.f_measured_clean= (1.0-X_factor)*self.intrinsic_frequency - X_factor*pulsars.ephemeris
        
        # measurement_noise = generator.normal(0, pulsars.σm,self.f_measured_clean.shape) # Measurement noise. Seeded
        # self.f_measured = self.f_measured_clean + measurement_noise


        # #add time as part of the data object
        # self.t = t