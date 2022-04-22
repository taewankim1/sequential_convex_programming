import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Landing2D(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT,linearization="numeric_central"):
        super().__init__(name,ix,iu,delT,linearization)
        self.m = 2
        self.I = 1e-2
        self.r_t = 1e-2
        self.g = 1
        
    def forward(self,x,u,idx=None,discrete=True):
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
        rx = x[:,0]
        ry = x[:,1]
        vx = x[:,2]
        vy = x[:,3]
        t = x[:,4]
        w = x[:,5]
        
        gimbal = u[:,0]
        thrust = u[:,1]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = vy
        f[:,2] = 1 / self.m * (-np.sin(t+gimbal)) * thrust
        f[:,3] = 1 / self.m * (np.cos(t+gimbal)) * thrust - self.g
        f[:,4] = w
        f[:,5] = 1 / self.I * (-np.sin(gimbal)*thrust*self.r_t)

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f
    
    # def diff(self,x,u):

    #     # dimension
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1
    #         x = np.expand_dims(x,axis=0)
    #         u = np.expand_dims(u,axis=0)
    #     else :
    #         N = np.size(x,axis = 0)
        
    #     # state & input
    #     x1 = x[:,0]
    #     x2 = x[:,1]
    #     x3 = x[:,2]
        
    #     v = u[:,0]
    #     w = u[:,1]    
        
    #     fx = np.zeros((N,self.ix,self.ix))
    #     fx[:,0,0] = 1.0
    #     fx[:,0,1] = 0.0
    #     fx[:,0,2] = - self.delT * v * np.sin(x3)
    #     fx[:,1,0] = 0.0
    #     fx[:,1,1] = 1.0
    #     fx[:,1,2] = self.delT * v * np.cos(x3)
    #     fx[:,2,0] = 0.0
    #     fx[:,2,1] = 0.0
    #     fx[:,2,2] = 1.0
        
    #     fu = np.zeros((N,self.ix,self.iu))
    #     fu[:,0,0] = self.delT * np.cos(x3)
    #     fu[:,0,1] = 0.0
    #     fu[:,1,0] = self.delT * np.sin(x3)
    #     fu[:,1,1] = 0.0
    #     fu[:,2,0] = 0.0
    #     fu[:,2,1] = self.delT
        
    #     return np.squeeze(fx) , np.squeeze(fu)