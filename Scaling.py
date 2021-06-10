import numpy as np
import time
import random
import IPython
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))


def compute_scaling(x,u) :
    # Scaling
    tol_zero = np.sqrt(np.finfo(float).eps)
    x_max = np.max(x,0)
    x_min = np.min(x,0)
    u_max = np.max(u,0)
    u_min = np.min(u,0)
    x_intrvl = [0,1]
    u_intrvl = [0,1]
    x_width = x_intrvl[1] - x_intrvl[0]
    u_width = u_intrvl[1] - u_intrvl[0]
    
    Sx = (x_max - x_min) / x_width
    Sx[np.where(Sx < tol_zero)] = 1
    Sx = np.diag(Sx)
    iSx = np.linalg.inv(Sx)
    sx = x_min - x_intrvl[0] * np.diag(Sx)

    Su = (u_max - u_min) / u_width
    Su[np.where(Su < tol_zero)] = 1
    Su = np.diag(Su)
    iSu = np.linalg.inv(Su)
    su = u_min - u_intrvl[0] * np.diag(Su)
    
    return Sx,iSx,sx,Su,iSu,su