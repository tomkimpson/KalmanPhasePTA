


import logging
logging.basicConfig()
logging.getLogger(name="KalmanGW").setLevel(logging.INFO)

"""
Class of parameters which define the system
"""
class SystemParameters:


    def __init__(self,
                 T       = 10,           # how long to integrate for in years
                 cadence = 7,            # the interval between observations
                 Ω       = 5.0e-7,       # GW angular frequency
                 Φ0      = 0.20,         # GW phase offset at t=0
                 ψ       =  2.50,        # GW polarisation angle
                 ι       = 1.0,          # GW source inclination
                 δ       =  1.0,         # GW source declination
                 α       =  1.0,         # GW source right ascension
                 h       =  5e-15,       # GW strain
                 process_noise = True,   # the process noise on the pulsars. Any of "True", "Fixed", "Random". See pulsars.py for example
                 σm = 1e-11,             # measurement noise standard deviation
                 Npsr = 0,               # Number of pulsars to use in PTA. 0 = all
                 use_psr_terms_in_data=True, # when generating the synthetic data, include pulsar terms?
                 measurement_model='pulsar', # what do you want the KF measurement model to be? One of pulsar, earth,null
                 seed = 1234,                # this is the noise seed. It is used for realisations of process noise and measurement noise and also if random pulsars or random process noise covariances are requested 
                 σp = 1e-20                  # only used if process_noise != True. Assign the process noise s.d. = σp for all pulsars if "Fixed". Assign randomly within U(σp/10,σp*10) if random. 
                 ): 

        logging.info("Welcome to the Kalman Filter Nested Sampler for PTA GW systems")

        self.T = T 
        self.cadence = cadence
        self.Ω = Ω
        self.Φ0 = Φ0
        self.ψ = ψ
        self.ι = ι
        self.δ = δ
        self.α = α
        self.h = h
        self.σp = σp #can be = None for random assignment. Handle NF conversion in pulsars.py

        self.σm = σm
        self.Npsr = int(Npsr)

        self.use_psr_terms_in_data = use_psr_terms_in_data 
        self.measurement_model = measurement_model
        self.seed = seed

        logging.info(f"Random seed is {self.seed}")


        
    

