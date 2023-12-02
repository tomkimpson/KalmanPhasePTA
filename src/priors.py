


import bilby
import numpy as np
import logging 
logging.getLogger().setLevel(logging.INFO)




"""
Main external function for defining priors
"""
def bilby_priors_dict(PTA,P,set_state_parameters_as_known=False,set_measurement_parameters_as_known=False):


    logging.info('Setting the bilby priors dict')


    init_parameters = {}
    priors = bilby.core.prior.PriorDict()
    
    #Measurement priors
    init_parameters,priors = _set_prior_on_measurement_parameters(init_parameters,priors,P,set_measurement_parameters_as_known) 

    #State priors
    init_parameters,priors = _set_prior_on_state_parameters(init_parameters,priors,PTA, set_state_parameters_as_known)
 
    #Measurement noise priors. Always known
    init_parameters["sigma_m"] = None
    priors["sigma_m"] = P.σm

    #return init_parameters,priors
    return priors







"""
Create a delta function prior about the true value
"""
def _add_to_bibly_priors_dict_constant(x,label,init_parameters,priors):


    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
        priors[key] = f
        i+= 1

    return init_parameters,priors


"""
Add  logarithmic prior vector
"""
def _add_to_bibly_priors_dict_log(x,label,init_parameters,priors,lower,upper): #same lower/upper for every one
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.LogUniform(lower,upper, key)
        
        i+= 1

    return init_parameters,priors


"""
Add uniform prior vector
"""
def _add_to_bibly_priors_dict_uniform(x,label,init_parameters,priors,tol):
    
    i = 0
    for f in x:
        key = label+str(i)
        init_parameters[key] = None
      
        priors[key] = bilby.core.prior.Uniform(f-np.abs(f*tol),f+ np.abs(f*tol), key)
        
        i+= 1

    return init_parameters,priors



"""
Set a prior on the state parameters
"""
def _set_prior_on_state_parameters(init_parameters,priors,PTA,set_parameters_as_known):

    if set_parameters_as_known:
        logging.info('Setting fully informative priors on PSR parameters')

        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.f,"f0",init_parameters,priors)     
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.fdot,"fdot",init_parameters,priors)    
        #init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.σp,"sigma_p",init_parameters,priors)  
        #init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.γ,"gamma",init_parameters,priors)           
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.d,"distance",init_parameters,priors) 


        #For now, just one gamma and sigma p
        init_parameters["gamma"] = None
        priors["gamma"] = PTA.γ[0]

        init_parameters["sigma_p"] = None
        priors["sigma_p"] = PTA.σp[0]
    
    else:
        logging.info('Setting uninformative priors on PSR parameters. NOTE: just a place holder and need updating to be accurate')
        #TODO

        init_parameters,priors = _add_to_bibly_priors_dict_uniform(PTA.f,"f0",init_parameters,priors,tol=1e-10)      #uniform
        init_parameters,priors = _add_to_bibly_priors_dict_uniform(PTA.fdot,"fdot",init_parameters,priors,tol=0.01) #uniform
        init_parameters,priors = _add_to_bibly_priors_dict_log(PTA.σp,"sigma_p",init_parameters,priors,1e-21,1e-19) #log. 
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.γ,"gamma",init_parameters,priors)           # constant
        init_parameters,priors = _add_to_bibly_priors_dict_constant(PTA.d,"d",init_parameters,priors) 


    return init_parameters,priors 






"""
Set a prior on the measurement parameters
"""
def _set_prior_on_measurement_parameters(init_parameters,priors,P,set_parameters_as_known):

    if set_parameters_as_known: #don't set a prior, just assume these are known exactly a priori

        logging.info('Setting fully informative priors on GW parameters')
        
        #Add all the GW quantities
        init_parameters[f"omega_gw"] = None
        priors[f"omega_gw"] = P.Ω
        print(P.Ω)

        init_parameters[f"phi0_gw"] = None
        priors[f"phi0_gw"] = P.Φ0

        init_parameters[f"psi_gw"] = None
        priors[f"psi_gw"] = P.ψ

        init_parameters[f"iota_gw"] = None
        priors[f"iota_gw"] = P.ι

        init_parameters[f"delta_gw"] = None
        priors[f"delta_gw"] = P.δ

        init_parameters[f"alpha_gw"] = None
        priors[f"alpha_gw"] = P.α

        init_parameters[f"h"] = None
        priors[f"h"] = P.h

    else:
        logging.info('Setting uninformative priors on GW parameters. NOTE: this needs updating')

            
        #Add all the GW quantities
        init_parameters[f"omega_gw"] = None
        priors[f"omega_gw"] = bilby.core.prior.Uniform(1e-7, 9e-7, 'omega_gw')


        init_parameters[f"phi0_gw"] = None
        priors[f"phi0_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'phi0_gw')

        init_parameters[f"psi_gw"] = None
        priors[f"psi_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'psi_gw')

        init_parameters[f"iota_gw"] = None
        priors[f"iota_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2.0, 'iota_gw')

        init_parameters[f"delta_gw"] = None
        priors[f"delta_gw"] = bilby.core.prior.Uniform(0.0, np.pi/2, 'delta_gw')

        init_parameters[f"alpha_gw"] = None
        priors[f"alpha_gw"] = bilby.core.prior.Uniform(0.0, np.pi, 'alpha_gw')

        init_parameters[f"h"] = None
        priors[f"h"] = bilby.core.prior.Uniform(1e-15, 9e-15, 'h')


    return init_parameters,priors 



