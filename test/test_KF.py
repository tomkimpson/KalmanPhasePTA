
    # print("Check posdef P output")
    # try:
    #     np.linalg.cholesky(Pnew)
    #     print("is pos def")
    # except:
    #     print("not pos def")




    # assert isDiag(S)






@njit(fastmath=True)
def isDiag(M):
    i, j = np.nonzero(M)
    return np.all(i == j)
