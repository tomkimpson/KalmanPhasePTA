import numba as nb
import numpy as np 

@nb.jit(nopython=True) #From https://stackoverflow.com/a/58017351
def block_diag_view_jit(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num * rows, num * cols), dtype=arr.dtype)
    for k in range(num):
        result[k * rows:(k + 1) * rows, k * cols:(k + 1) * cols] = arr
    return result



@nb.jit(parallel=True)
def block_diag_loop_jit(arr, num):
    rows, cols = arr.shape
    result = np.zeros((num * rows, num * cols), dtype=arr.dtype)
    for k in nb.prange(num):
        for i in nb.prange(rows):
            for j in nb.prange(cols):
                result[i + (rows * k), j + (cols * k)] = arr[i, j]
    return result