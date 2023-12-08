import bilby
import logging
logger = logging.getLogger(__name__).setLevel(logging.INFO)

import sys
"""Here is some test documentation"""
class BilbyLikelihood(bilby.Likelihood):

    def __init__(self,KalmanModel,parameters):

        super().__init__(parameters=parameters)
        self.model = KalmanModel
        
    def log_likelihood(self):
        ll = self.model.likelihood(self.parameters)
        return ll
    
            
def BilbySampler(KalmanFilter,init_parameters,priors,NS_settings):
   
    
    likelihood = BilbyLikelihood(KalmanFilter,init_parameters)

    #Run the sampler
    logging.info("Starting the bilby sampler")
    result = bilby.run_sampler(likelihood, priors, 
                              label            =NS_settings.label,
                              outdir           =NS_settings.outdir,
                              sampler          =NS_settings.sampler,
                              sample           =NS_settings.sample,
                              bound            =NS_settings.bound,
                              check_point_plot =False,
                              npoints          =NS_settings.npoints,
                              dlogz            =NS_settings.dlogz,
                              npool            =NS_settings.npool,
                              plot             =NS_settings.plot,
                              resume           =NS_settings.resume)

    return result