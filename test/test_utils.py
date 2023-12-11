#This is the test file for py_src/gravitational_waves.py 
from src import utils
import numpy as np 


"""Check that our function for constructing large matrices from blocks works as expected"""
def test_block_construction():


    Npsr = 2
  
    component_array = np.array([[1,2],
                                [0,3]]) 

    F = utils.block_diag_view_jit(component_array,Npsr) 


    Ftrue = np.array([[1,2,0,0],
                      [0,3,0,0],
                      [0,0,1,2],
                      [0,0,0,3]])

    assert np.all(F == Ftrue)


    
    


    #And some different matrix just to be sure
    Npsr = 2
  
    component_array = np.array([[7,12],
                                [8,13]]) 

    F = utils.block_diag_view_jit(component_array,Npsr) 


    Ftrue = np.array([[7,12,0,0],
                      [8,13,0,0],
                      [0,0,7,12],
                      [0,0,8,13]])

    assert np.all(F == Ftrue)




    #And Npsr = 3
    Npsr = 3
  
    component_array = np.array([[7,12],
                                [8,13]]) 

    F = utils.block_diag_view_jit(component_array,Npsr) 


    Ftrue = np.array([[7,12,0,0,0,0],
                      [8,13,0,0,0,0],
                      [0,0,7,12,0,0],
                      [0,0,8,13,0,0],
                      [0,0,0,0,7,12],
                      [0,0,0,0,8,13]])

    assert np.all(F == Ftrue)
