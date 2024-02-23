#This is the test file for py_src/gravitational_waves.py 
from src import system_parameters
# import random 
import numpy as np 
# from numpy import sin,cos

"""Check that we can call SystemParameters and the variables are generally set properly"""
def test_basic_call():

    N = 5
    
    #Just check the GW parameters
    Ω = np.random.uniform(low=0.0,high=2*np.pi,size=N)
    Φ0= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ψ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    ι= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    δ= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    α= np.random.uniform(low=0.0,high=2*np.pi,size=N)
    h= np.random.uniform(low=0.0,high=2*np.pi,size=N)

    #And observation time
    T = np.random.uniform(1,10,size=N)

    for i in range(N):
        P   = system_parameters.SystemParameters(Ω=Ω[i],Φ0=Φ0[i],ψ=ψ[i],ι=ι[i],δ=δ[i],α=α[i],h=h[i],T=T[i]) 
        assert P.Ω==Ω[i]
        assert P.Φ0==Φ0[i]
        assert P.ψ==ψ[i]
        assert P.ι==ι[i]
        assert P.δ==δ[i]
        assert P.α==α[i]
        assert P.h==h[i]
        assert P.T==T[i]


"""Make sure that the booleans behave as expected"""
def test_booleans():


    NS_settings = system_parameters.NestedSamplerSettings(resume='True', plot = 'True')
    assert NS_settings.resume
    assert NS_settings.plot

    NS_settings = system_parameters.NestedSamplerSettings(resume='False', plot = 'False')
    assert not NS_settings.resume
    assert not NS_settings.plot

    NS_settings = system_parameters.NestedSamplerSettings(resume='True', plot = 'False')
    assert NS_settings.resume
    assert not NS_settings.plot