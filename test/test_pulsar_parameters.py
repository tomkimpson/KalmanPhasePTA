#This is the test file for py_src/gravitational_waves.py 
from src import system_parameters, pulsars
# import random 
import numpy as np 
# from numpy import sin,cos

"""Check that the values returned for the pulsar parameters are as expected"""
def test_reasonable_pulsar_values():


    P   = system_parameters.SystemParameters() 
    PTA = pulsars.Pulsars(P)                                       # setup the PTA

    #Check the range of f values
    assert np.all(PTA.f < 1000) #No pulsars should be above 1000 Hz
    assert ~np.all(PTA.f < 50) #or slower than 50Hz


    #Check the range of fdots 
    assert np.all(np.abs(PTA.fdot) < 1e-13) #spindowns should be small 


    #Check distances
    #All pulsars should be in range 0.1-7 kpc, after converting units
    c = 3e8
    pc = 3e16
    assert 0.1 <= np.all(PTA.d*c/pc) <= 7


    #Check all gammas 
    assert np.all(PTA.γ == 1e-13)

    #Check all alphas, deltas in range 
    assert np.all(PTA.δ <= np.pi/2)
    assert np.all(PTA.δ >= -np.pi/2)

    assert np.all(PTA.α <= 2*np.pi)
    assert np.all(PTA.α >= 0)




def test_unit_vector():

    N = 5
    delta = np.random.uniform(low=-np.pi/2,high=np.pi/2,size=N)
    alpha = np.random.uniform(low=0.0,high=2*np.pi,size=N)


    m = pulsars._unit_vector(np.pi/2.0 - delta,alpha)
    for i in range(N):
        np.testing.assert_almost_equal(np.linalg.norm(m[i]), 1.0)


    #When delta is in the plane, z component should = 0
    delta = np.array([0.0])
    alpha = np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m = pulsars._unit_vector(np.pi/2.0 - delta,alpha)
    np.testing.assert_almost_equal(m[:,2], 0)

    #When delta is maximal z component should =1
    delta = np.array([np.pi/2])
    alpha = np.random.uniform(low=0.0,high=2*np.pi,size=1)
    m = pulsars._unit_vector(np.pi/2.0 - delta,alpha)
    np.testing.assert_almost_equal(m[:,2], 1)


    #Also check that if we call Pulsars(P), the output q vector is also a unit vector
    P   = system_parameters.SystemParameters() 
    PTA = pulsars.Pulsars(P)  
    q = PTA.q
    for i in range(len(q)):
        np.testing.assert_almost_equal(np.linalg.norm(q[i]), 1.0)






