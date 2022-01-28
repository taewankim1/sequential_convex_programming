import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class AircraftKinematics(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT,linearization="numeric_central"):
        super().__init__(name,ix,iu,delT,linearization)
        self.m = 288938
        # self.m = 1

        
    def forward(self,x,u,idx=None,discrete=False):
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
        rz = x[:,2]
        v = x[:,3] # speed
        gamma = x[:,4] # path angle
        psi = x[:,5] # velocity heading
        
        # gamma = u[:,0] 
        gamma_dot = u[:,0] 
        psi_dot = u[:,1] 
        thrust = u[:,2] # thrust
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(gamma) * np.cos(psi)
        f[:,1] = v * np.cos(gamma) * np.sin(psi)
        f[:,2] = v * np.sin(gamma)
        f[:,3] = 1 / self.m * thrust
        f[:,4] = gamma_dot
        f[:,5] = psi_dot


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