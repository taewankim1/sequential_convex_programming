import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Aircraft3dofNondimension(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearization="numeric_central",beta=600):
        super().__init__(name,ix,iu,linearization)
        self.m = 288938
        self.g = 9.81
        self.Sw = 510.97
        self.CD0 = 0.022
        self.K = 0.045
        self.rho = 1.225

        # normalized parameter
        self.beta = beta
        self.rho_bar = (self.rho * (self.g**2)) / (2*self.m*self.g*(self.beta**2)/self.Sw)
        
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
        
        # output
        f = np.zeros(x.shape)
        f[:,0] = v * np.cos(gamma) * np.cos(psi)
        f[:,1] = v * np.cos(gamma) * np.sin(psi)
        f[:,2] = v * np.sin(gamma)
        f[:,3] = thrust - self.rho_bar * v * v * (self.CD0 + self.K * CL * CL) - np.sin(gamma)
        f[:,4] = self.rho_bar * v * CL * np.cos(phi) - np.cos(gamma) / v
        f[:,5] = - self.rho_bar * v * CL * np.sin(phi) / np.cos(gamma)

        return f
    
    # def diff(self,x,u,discrete=True) :
    #     assert discrete == False
    #     # state & input size
    #     ix = self.ix
    #     iu = self.iu
        
    #     ndim = np.ndim(x)
    #     if ndim == 1: # 1 step state & input
    #         N = 1
    #         x = np.expand_dims(x,axis=0)
    #         u = np.expand_dims(u,axis=0)
    #     else :
    #         N = np.size(x,axis = 0)
    #     # state & input
    #     rx = x[:,0]
    #     ry = x[:,1]
    #     rz = x[:,2]
    #     v = x[:,3] # speed
    #     gamma = x[:,4] # path angle
    #     psi = x[:,5] # velocity heading
        
    #     CL = u[:,0] # lift coefficient
    #     phi = u[:,1] # bank angle
    #     thrust = u[:,2] # thrust

    #     m,g,Sw,CD0,K = self.m,self.g,self.Sw,self.CD0,self.K

    #     fx = np.zeros((N,ix,ix))
    #     fx[:,0,3] = np.cos(gamma)*np.cos(psi)
    #     fx[:,0,4] = -v*np.sin(gamma)*np.cos(psi)
    #     fx[:,0,5] = -v*np.sin(psi)*np.cos(gamma)
    #     fx[:,1,3] = np.sin(psi)*np.cos(gamma)
    #     fx[:,1,4] = -v*np.sin(gamma)*np.sin(psi)
    #     fx[:,1,5] = v*np.cos(gamma)*np.cos(psi)
    #     fx[:,2,3] = np.sin(gamma)
    #     fx[:,2,4] = v*np.cos(gamma)
    #     fx[:,3,2] = (0.00600217215675481*Sw*v**2*(1 - 2.25237731658222e-5*rz)**4.256*(CD0 + CL**2*K)/(82.667366 - 0.001861981*rz) - 1.38139853548573e-5*Sw*v**2*(1 - 2.25237731658222e-5*rz)**5.256*(CD0 + CL**2*K)/(1 - 2.25237731658222e-5*rz)**2)/m
    #     fx[:,3,3] = -101.400930904549*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*(CD0 + CL**2*K)/(m*(82.667366 - 0.001861981*rz))
    #     fx[:,3,4] = -g*np.cos(gamma)
    #     fx[:,4,2] = (-0.00600217215675481*CL*Sw*v**2*(1 - 2.25237731658222e-5*rz)**4.256*np.cos(phi)/(82.667366 - 0.001861981*rz) + 1.38139853548573e-5*CL*Sw*v**2*(1 - 2.25237731658222e-5*rz)**5.256*np.cos(phi)/(1 - 2.25237731658222e-5*rz)**2)/(m*v)
    #     fx[:,4,3] = 101.400930904549*CL*Sw*(1 - 2.25237731658222e-5*rz)**5.256*np.cos(phi)/(m*(82.667366 - 0.001861981*rz)) - (50.7004654522744*CL*Sw*v**2*(1 - 2.25237731658222e-5*rz)**5.256*np.cos(phi)/(82.667366 - 0.001861981*rz) - g*m*np.cos(gamma))/(m*v**2)
    #     fx[:,4,4] = g*np.sin(gamma)/v
    #     fx[:,5,2] = 0.00600217215675481*CL*Sw*v*(1 - 2.25237731658222e-5*rz)**4.256*np.sin(phi)/(m*(82.667366 - 0.001861981*rz)*np.cos(gamma)) - 1.38139853548573e-5*CL*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.sin(phi)/(m*(1 - 2.25237731658222e-5*rz)**2*np.cos(gamma))
    #     fx[:,5,3] = -50.7004654522744*CL*Sw*(1 - 2.25237731658222e-5*rz)**5.256*np.sin(phi)/(m*(82.667366 - 0.001861981*rz)*np.cos(gamma))
    #     fx[:,5,4] = -50.7004654522744*CL*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.sin(gamma)*np.sin(phi)/(m*(82.667366 - 0.001861981*rz)*np.cos(gamma)**2)


    #     fu = np.zeros((N,ix,iu))
        
    #     fu[:,3,0] = -101.400930904549*CL*K*Sw*v**2*(1 - 2.25237731658222e-5*rz)**5.256/(m*(82.667366 - 0.001861981*rz))
    #     fu[:,3,2] = 1/m
    #     fu[:,4,0] = 50.7004654522744*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.cos(phi)/(m*(82.667366 - 0.001861981*rz))
    #     fu[:,4,1] = -50.7004654522744*CL*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.sin(phi)/(m*(82.667366 - 0.001861981*rz))
    #     fu[:,5,0] = -50.7004654522744*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.sin(phi)/(m*(82.667366 - 0.001861981*rz)*np.cos(gamma))
    #     fu[:,5,1] = -50.7004654522744*CL*Sw*v*(1 - 2.25237731658222e-5*rz)**5.256*np.cos(phi)/(m*(82.667366 - 0.001861981*rz)*np.cos(gamma))

    #     return fx.squeeze(),fu.squeeze()