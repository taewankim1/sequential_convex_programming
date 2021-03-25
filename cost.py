
# coding: utf-8

# In[1]:

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
class OptimalcontrolCost(object) :
    def __init__(self,name) :
        self.name = name

    def estimate_cost(self) :
        print("this is in parent class")
        pass

    # def diff_cost(self,x,u):
        
    #     # state & input size
    #     ix = self.ix
    #     iu = self.iu
        
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1

    #     else :
    #         N = np.size(x,axis = 0)

    #     # numerical difference
    #     h = pow(2,-17)
    #     eps_x = np.identity(ix)
    #     eps_u = np.identity(iu)

    #     # expand to tensor
    #     # print_np(x)
    #     x_mat = np.expand_dims(x,axis=2)
    #     u_mat = np.expand_dims(u,axis=2)

    #     # diag
    #     x_diag = np.tile(x_mat,(1,1,ix))
    #     u_diag = np.tile(u_mat,(1,1,iu))

    #     # augmented = [x_diag x], [u, u_diag]
    #     x_aug = x_diag + eps_x * h
    #     x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
    #     x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))
        
    #     u_aug = u_diag + eps_u * h
    #     u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
    #     u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

    #     # numerical difference
    #     c_nominal = self.estimate_cost(x,u)
    #     c_change = self.estimate_cost(x_aug,u_aug)
    #     c_change = np.reshape(c_change,(N,1,iu+ix))

    #     c_diff = ( c_change - np.reshape(c_nominal,(N,1,1)) ) / h
    #     c_diff = np.reshape(c_diff,(N,iu+ix))
            
    #     return  np.squeeze(c_diff)
    
    # def hess_cost(self,x,u):
        
    #     # state & input size
    #     ix = self.ix
    #     iu = self.iu
        
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1

    #     else :
    #         N = np.size(x,axis = 0)
        
    #     # numerical difference
    #     h = pow(2,-17)
    #     eps_x = np.identity(ix)
    #     eps_u = np.identity(iu)

    #     # expand to tensor
    #     x_mat = np.expand_dims(x,axis=2)
    #     u_mat = np.expand_dims(u,axis=2)

    #     # diag
    #     x_diag = np.tile(x_mat,(1,1,ix))
    #     u_diag = np.tile(u_mat,(1,1,iu))

    #     # augmented = [x_diag x], [u, u_diag]
    #     x_aug = x_diag + eps_x * h
    #     x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
    #     x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

    #     u_aug = u_diag + eps_u * h
    #     u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
    #     u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))


    #     # numerical difference
    #     c_nominal = self.diff_cost(x,u)
    #     c_change = self.diff_cost(x_aug,u_aug)
    #     c_change = np.reshape(c_change,(N,iu+ix,iu+ix))
    #     c_hess = ( c_change - np.reshape(c_nominal,(N,1,ix+iu)) ) / h
    #     c_hess = np.reshape(c_hess,(N,iu+ix,iu+ix))
         
    #     return np.squeeze(c_hess)


class unicycle(OptimalcontrolCost):
    def __init__(self,name,x_t,N):
        super().__init__(name)
       
        self.Q = 0*np.identity(3)
        # self.Q = 1e-1 * self.Q
        # self.Q[2,2] = 1e-2 * self.Q[2,2]

        self.R = 10 * np.identity(2)
        
        self.ix = 3
        self.iu = 2
        self.x_t = x_t #np.tile(x_t,(N,1))
        self.N = N

    def estimate_cost(self,x,u):
        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
            
        x_diff = np.copy(x)
        x_diff[:,0] = x_diff[:,0] - self.x_t[0]
        x_diff[:,1] = x_diff[:,1] - self.x_t[1]
        x_diff[:,2] = x_diff[:,2] - self.x_t[2]

        x_mat = np.expand_dims(x_diff,2)
        Q_mat = np.tile(self.Q,(N,1,1))
        lx = np.squeeze(np.matmul(np.matmul(np.transpose(x_mat,(0,2,1)),Q_mat),x_mat))
        
        # cost for input
        u_mat = np.expand_dims(u,axis=2)
        R_mat = np.tile(self.R,(N,1,1))
        lu = np.squeeze( np.matmul(np.matmul(np.transpose(u_mat,(0,2,1)),R_mat),u_mat) )
        
        cost_total = 0.5*(lx + lu)
        
        return cost_total
        
    def estimate_cost_cvx(self,x,u):
        # dimension
        cost_total = 0.5*(cp.quad_form(x-self.x_t, self.Q) + cp.quad_form(u,self.R))
        
        return cost_total
    
    
