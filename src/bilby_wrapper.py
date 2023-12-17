import bilby
import logging
logger = logging.getLogger(__name__).setLevel(logging.INFO)
import os
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


    #In practice I find that the resume argument does not work properly with Bilby
    #If a resume file exists, it uses it, regardless of whether we set resume=True/False
    #But what if we want to overwrite any existing file that has the same label?
    #We handle that by deleting the resume file, which is not ideal, but works OK in practice. 

    resume_file = NS_settings.outdir + NS_settings.label+'_resume.pickle'
    
    if not NS_settings.resume: #if we are not resuming
        if os.path.exists(resume_file): #but a file exists
            logging.info(f"Deleting existing resume file: {resume_file}")
            os.remove(resume_file) #delete that resume file
    else:
        logging.info(f"Existing resume file {resume_file} is being used to commence the run")



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