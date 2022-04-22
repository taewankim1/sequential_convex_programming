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

class AircraftKinematics(OptimalcontrolCost):
    def __init__(self,name,ix,iu,N):
        super().__init__(name,ix,iu,N)
       
        self.Q = 0.0*np.identity(self.ix)

        self.R = np.identity(self.iu)
        self.R[0,0] = 1e3 
        self.R[1,1] = 1e3 
        self.R[2,2] = 1e-8 
        # self.R[2,2] = 1.0

    def estimate_cost(self,x,u):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)

        x_mat = np.expand_dims(x,2)
        Q_mat = np.tile(self.Q,(N,1,1))
        lx = np.squeeze(np.matmul(np.matmul(np.transpose(x_mat,(0,2,1)),Q_mat),x_mat))
        
        # cost for input
        u_mat = np.expand_dims(u,axis=2)
        R_mat = np.tile(self.R,(N,1,1))
        lu = np.squeeze( np.matmul(np.matmul(np.transpose(u_mat,(0,2,1)),R_mat),u_mat) )
        
        cost_total = 0.5*(lx + lu)
        
        return cost_total

    def estimate_final_cost(self,x,u) :
        return self.estimate_cost(x,u)
        
    def estimate_cost_cvx(self,x,u,idx=0):
        # dimension
        cost_total = 0.5*(cp.quad_form(x, self.Q) + cp.quad_form(u,self.R))
        
        return cost_total