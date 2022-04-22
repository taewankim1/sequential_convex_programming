from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
import cvxpy as cp

def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from cost import OptimalcontrolCost

class Aircraft3dof(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)

    def estimate_cost(self,x,u):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        cost_total = u[:,2]
        return cost_total

    def estimate_final_cost(self,x,u) :
        return self.estimate_cost(x,u)
        
    def estimate_cost_cvx(self,x,u,idx=0):
        cost_total = u[2]
        return cost_total