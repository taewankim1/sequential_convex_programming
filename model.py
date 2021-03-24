
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


class OptimalcontrolModel(object) :
    def __init__(self,name,ix,iu,delT) :
        self.name = name
        self.ix = ix
        self.iu = iu
        self.delT = delT

    def forward(self) :
        print("this is in parent class")
        pass

    def diff(self) :
        print("this is in parent class")
        pass

    def diff_numeric(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            # x = np.expand_dims(x,axis=0)
            # u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # numerical difference
        h = pow(2,-17)
        eps_x = np.identity(ix)
        eps_u = np.identity(iu)

        # expand to tensor
        x_mat = np.expand_dims(x,axis=2)
        u_mat = np.expand_dims(u,axis=2)

        # diag
        x_diag = np.tile(x_mat,(1,1,ix))
        u_diag = np.tile(u_mat,(1,1,iu))

        # augmented = [x_aug x], [u, u_aug]
        x_aug = x_diag + eps_x * h
        x_aug = np.dstack((x_aug,np.tile(x_mat,(1,1,iu))))
        x_aug = np.reshape( np.transpose(x_aug,(0,2,1)), (N*(iu+ix),ix))

        u_aug = u_diag + eps_u * h
        u_aug = np.dstack((np.tile(u_mat,(1,1,ix)),u_aug))
        u_aug = np.reshape( np.transpose(u_aug,(0,2,1)), (N*(iu+ix),iu))

        # numerical difference
        f_nominal = self.forward(x,u,0) 
        f_change = self.forward(x_aug,u_aug,0)
        # print_np(f_change)
        f_change = np.reshape(f_change,(N,ix+iu,ix))
        # print_np(f_nominal)
        # print_np(f_change)
        f_diff = ( f_change - np.reshape(f_nominal,(N,1,ix)) ) / h
        # print_np(f_diff)
        f_diff = np.transpose(f_diff,[0,2,1])
        fx = f_diff[:,:,0:ix]
        fu = f_diff[:,:,ix:ix+iu]
        
        return np.squeeze(fx), np.squeeze(fu)

class unicycle(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT):
        super().__init__(name,ix,iu,delT)
        
    def forward(self,x,u,idx,discrete=True):
        
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(x3)
        f[:,1] = v * np.sin(x3)
        f[:,2] = w

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f
    
    def diff(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]    
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 1.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = - self.delT * v * np.sin(x3)
        fx[:,1,0] = 0.0
        fx[:,1,1] = 1.0
        fx[:,1,2] = self.delT * v * np.cos(x3)
        fx[:,2,0] = 0.0
        fx[:,2,1] = 0.0
        fx[:,2,2] = 1.0
        
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = self.delT * np.cos(x3)
        fu[:,0,1] = 0.0
        fu[:,1,0] = self.delT * np.sin(x3)
        fu[:,1,1] = 0.0
        fu[:,2,0] = 0.0
        fu[:,2,1] = self.delT
        
        return np.squeeze(fx) , np.squeeze(fu)
    
    
class SimpleLinear(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT):
        super().__init__(name,ix,iu,delT)
        
    def forward(self,x,u,idx,discrete=True):
        
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)
     
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        
        v = u[:,0]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = x2
        f[:,1] = v

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f
    
    def diff(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        
        v = u[:,0]
        
        fx = np.zeros((N,self.ix,self.ix))
        fx[:,0,0] = 1.0
        fx[:,0,1] = self.delT
        fx[:,1,0] = 0.0
        fx[:,1,1] = 1.0
        
        fu = np.zeros((N,self.ix,self.iu))
        fu[:,0,0] = 0.0
        fu[:,1,0] = self.delT
        
        return np.squeeze(fx) , np.squeeze(fu)