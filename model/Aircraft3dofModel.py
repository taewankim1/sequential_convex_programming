import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Aircraft3dof(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearization="numeric_central"):
        super().__init__(name,ix,iu,linearization)
        self.m = 288938
        self.g = 9.81
        self.Sw = 510.97
        self.CD0 = 0.022
        self.K = 0.045
        
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
        
        CL = u[:,0] # lift coefficient
        phi = u[:,1] # bank angle
        thrust = u[:,2] # thrust

        # density
        def get_density(rz) :
            flag_1 = rz < 11000 
            T1 = 15.04 - 0.00649 * rz # celsius
            p1 = 101.29 * np.power((T1+273.1)/288.08,5.256)
            rho1 = p1 / (0.2869 * (T1 + 273.1))

            flag_2 = np.logical_and(rz >= 11000, rz<25000)
            T2 = -56.46 # not used
            p2 = 22.65 * np.exp(1.73-0.000157 * rz)
            rho2 = p2 / (0.2869 * (T2 + 273.1))

            flag_3 = rz >= 25000
            T3 = -131.21 + 0.00299 * rz
            p3 = 2.488 * np.power((T1+273.1)/216.6,-11.388)
            rho3 = p3 / (0.2869 * (T3 + 273.1))
            return rho1*flag_1 + rho2*flag_2 + rho3*flag_3
        rho = get_density(rz)
        # rho = 1.225

        # Lift & drag force
        L = 0.5 * rho * v * v * self.Sw * CL
        D = 0.5 * rho * v * v * self.Sw * (self.CD0 + self.K  * CL * CL)
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(gamma) * np.cos(psi)
        f[:,1] = v * np.cos(gamma) * np.sin(psi)
        f[:,2] = v * np.sin(gamma)
        f[:,3] = 1 / self.m * (thrust - D - self.m * self.g * np.sin(gamma))
        f[:,4] = 1 /(self.m * v) * (L * np.cos(phi) - self.m * self.g * np.cos(gamma)) 
        f[:,5] = - L * np.sin(phi) / (self.m * v * np.cos(gamma))

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