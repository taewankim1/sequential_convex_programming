import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Reentry(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearization="numeric_central"):
        super().__init__(name,ix,iu,linearization)
        self.m = 907.186
        self.ge = 9.806
        self.Sref = 0.484
        self.rhoe = 1.225
        self.Re = 6371*1000
        self.bet = 0.14*1e-3
        self.CLstr = 0.45
        self.Estr = 3.24
        self.Bcnst = 0.5*self.rhoe*self.Re*self.Sref*self.CLstr/self.m
        self.scl_t = np.sqrt(self.Re/self.ge)
        self.scl_d = self.Re
        self.scl_v = np.sqrt(self.ge*self.Re)
        
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
        rh = x[:,2]
        v = x[:,3]
        gamma = x[:,4] 
        theta = x[:,5] 
        
        lam = u[:,0] 
        sig = u[:,1] 
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(theta)
        f[:,1] = v * np.sin(theta)
        f[:,2] = v * gamma
        f[:,3] = -0.5 * (self.Bcnst * v * v * np.exp(-self.bet * self.Re * rh)* (1+lam * lam))/self.Estr
        f[:,4] = self.Bcnst * v * np.exp(-self.bet * self.Re * rh) * lam * np.cos(sig) - 1 / v + v
        f[:,5] = self.Bcnst * v * np.exp(-self.bet * self.Re * rh) * lam * np.sin(sig)

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