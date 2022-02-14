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
        self.m = 288938
        self.g = 9.81
        self.Sw = 510.97
        self.CD0 = 0.022
        self.K = 0.045
        
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
        rz = x[:,2]
        v = x[:,3] # speed
        gamma = x[:,4] # path angle
        psi = x[:,5] # velocity heading
        
        CL = u[:,0] # lift coefficient
        phi = u[:,1] # bank angle
        thrust = u[:,2] # thrust

        # density
        def get_density(rz) :
            # flag_1 = rz < 11000 
            T1 = 15.04 - 0.00649 * rz # celsius
            p1 = 101.29 * np.power((T1+273.1)/288.08,5.256)
            rho1 = p1 / (0.2869 * (T1 + 273.1))

            # flag_2 = np.logical_and(rz >= 11000, rz<25000)
            # T2 = -56.46 # not used
            # p2 = 22.65 * np.exp(1.73-0.000157 * rz)
            # rho2 = p2 / (0.2869 * (T2 + 273.1))

            # flag_3 = rz >= 25000
            # T3 = -131.21 + 0.00299 * rz
            # p3 = 2.488 * np.power((T1+273.1)/216.6,-11.388)
            # rho3 = p3 / (0.2869 * (T3 + 273.1))
            # return rho1*flag_1 + rho2*flag_2 + rho3*flag_3
            return rho1
        # rho = get_density(rz)
        rho = 1.225

        # Lift & drag force
        L = 0.5 * rho * v * v * self.Sw * CL
        # D = 0.5 * rho * v * v * self.Sw * (self.CD0 + self.K  * CL * CL)
        D = 0.5 * rho * v * v * self.Sw * (self.CD0 + self.K  * 2 * CL)
        
        # output
        f = np.zeros_like(x)
        f[:,0] = v * np.cos(psi)
        f[:,1] = v * np.sin(psi)
        f[:,2] = v * gamma
        f[:,3] = 1 / self.m * (thrust - D - self.m * self.g * gamma)
        f[:,4] = 1 /(self.m * v) * (L - self.m * self.g ) 
        f[:,5] = - L * phi / (self.m * v)


        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f
    
