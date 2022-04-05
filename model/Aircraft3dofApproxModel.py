import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Aircraft3dofApproxModel(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearization="numeric_central"):
        super().__init__(name,ix,iu,linearization)
        self.m = 49940
        self.g = 9.8
        self.Sw = 112
        self.CD0 = 0.0197
        self.K = 0.0459
        self.T_max = 137.81 * 1e3
        self.CLalpha = 4.2
        self.CL0 = 0.4225
        
    def forward(self,x,u,idx=None):
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
        
        alpha = u[:,0] # angle of attack
        phi = u[:,1] # bank angle
        thrust = self.T_max * u[:,2] # thrust

        rho = 1.225

        # Lift & drag force
        CL = self.CL0 + self.CLalpha * alpha
        L = 0.5 * rho * v * v * self.Sw * CL
        D = 0.5 * rho * v * v * self.Sw * (self.CD0 + self.K  * CL * CL) 


        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(gamma) * np.cos(psi)
        f[:,1] = v * np.cos(gamma) * np.sin(psi)
        f[:,2] = v * np.sin(gamma)
        f[:,3] = 1 / self.m * (thrust * np.cos(alpha) - D - self.m * self.g * np.sin(gamma))
        f[:,4] = 1 /(self.m * v) * (thrust * np.sin(alpha) + L * np.cos(phi) - self.m * self.g * np.cos(gamma) ) 
        f[:,5] = - L * np.sin(phi) / (self.m * v * np.cos(gamma))
        # # output
        # f = np.zeros_like(x)
        # f[:,0] = v * np.cos(psi)
        # f[:,1] = v * np.sin(psi)
        # f[:,2] = v * gamma
        # f[:,3] = 1 / self.m * (thrust - D - self.m * self.g * gamma)
        # f[:,4] = 1 /(self.m * v_) * (L - self.m * self.g ) 
        # f[:,5] = - L * phi / (self.m * v_)

        return f

    
