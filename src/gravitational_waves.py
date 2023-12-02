from numpy import sin,cos 
import numpy as np 
from numba import jit,njit,prange
import sys


"""
Return the two polarisation tensors e_+, e_x
Reshapes allow vectorisation and JIT compatability 
Todo: check performance of explicit JIT loops
"""
#@njit(fastmath=True)
def _polarisation_tensors(m, n):



    # For e_+,e_x, Tensordot might be a bit faster, but list comprehension has JIT support
    # Note these are 1D arrays, rather than the usual 2D struture
    #todo: check these for speed up
    e_plus              = np.array([m[i]*m[j]-n[i]*n[j] for i in range(3) for j in range(3)]) 
    e_cross             = np.array([m[i]*n[j]-n[i]*m[j] for i in range(3) for j in range(3)])

    return e_plus,e_cross




#@njit(fastmath=True)
def principal_axes(theta,phi,psi):
    
    m1 = sin(phi)*cos(psi) - sin(psi)*cos(phi)*cos(theta)
    m2 = -(cos(phi)*cos(psi) + sin(psi)*sin(phi)*cos(theta))
    m3 = sin(psi)*sin(theta)
    m = np.array([m1,m2,m3])

    n1 = -sin(phi)*sin(psi) - cos(psi)*cos(phi)*cos(theta)
    n2 = cos(phi)*sin(psi) - cos(psi)*sin(phi)*cos(theta)
    n3 = cos(psi)*sin(theta)
    n = np.array([n1,n2,n3])

    return m,n









"""
Get the hplus and hcross amplitudes
"""
#@njit(fastmath=True)
def _h_amplitudes(h,ι): 
    return h*(1.0 + cos(ι)**2),h*(-2.0*cos(ι)) #hplus,hcross



# """
# This function is used to add two 2D matrices of different shapes
# a(K,T)
# b(K,N) 

# It returns an array of shape (K,T,N)
# """
# #@njit
# def add_matrices(a, b):
#     K, T, N = a.shape[0], a.shape[1], b.shape[1]
#     return a.reshape(K,T,1) + b.reshape(K,1,N)






def _prefactors(delta,alpha,psi,q,q_products,h,iota,omega):

    #Time -independent terms
    m,n                 = principal_axes(np.pi/2.0 - delta,alpha,psi) # Get the principal axes of the GW
    gw_direction        = np.cross(m,n)                               # The GW source direction. #todo: probably fast to have this not as a cross product - use cross product in unit test
    e_plus,e_cross      = _polarisation_tensors(m.T,n.T)              # The polarization tensors. Shape (3,3,K)
    hp,hx               = _h_amplitudes(h,iota)                       # plus and cross amplitudes. Shape (K,)
    Hij                 = hp * e_plus + hx * e_cross                  # amplitude tensor. Shape (3,3,K)
    H                   = np.dot(Hij,q_products)                      
    dot_product         = 1.0 + q @ gw_direction
  
    
    prefactor = -H/(2*omega*dot_product)
    return prefactor,dot_product








"""
What is the GW modulation factor, including all pulsar terms?
"""
#@njit(fastmath=True)
def gw_psr_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,d):
    prefactor,dot_product = _prefactors(delta,alpha,psi,q,q_products,h,iota,omega)

    omega_t = -omega*t
    omega_t = omega_t[:,None] #Reshape to (T,1) to allow broadcasting. #todo, setup everything as 2d automatically


    earth_term = np.sin(omega_t + phi0)
    pulsar_term = np.sin(omega_t + phi0+omega*dot_product*d)


    return prefactor*(earth_term - pulsar_term)
   
  

def gw_earth_terms(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,d):
    prefactor,dot_product = _prefactors(delta,alpha,psi,q,q_products,h,iota,omega)

    omega_t = -omega*t
    omega_t = omega_t[:,None] #Reshape to (T,1) to allow broadcasting. #todo, setup everything as 2d automatically

    earth_term = np.sin(omega_t + phi0)


    return prefactor*(earth_term)




"""
The null model - i.e. no GW
"""
#@njit(fastmath=True)
def null_model(delta,alpha,psi,q,q_products,h,iota,omega,t,phi0,d):
    return np.zeros((len(t),len(q))) #if there is no GW, the GW factor = 0.0
    




