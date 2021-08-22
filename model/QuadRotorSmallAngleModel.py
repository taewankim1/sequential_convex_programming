
# coding: utf-8

# In[ ]:

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel

class quadrotorsa(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT,linearization="numeric_central"):
        super().__init__(name,ix,iu,delT,linearization)
        self.g = 9.81
        
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
        roll = x[:,3]
        pitch = x[:,4]
        yaw = x[:,5]
        xdot = x[:,6]
        ydot = x[:,7]
        zdot = x[:,8]
        rolldot = x[:,9]
        pitchdot = x[:,10]
        yawdot = x[:,11]
        
        thrust = u[:,0]
        Mx = u[:,1]
        My = u[:,2]
        Mz = u[:,3]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = xdot
        f[:,1] = ydot
        f[:,2] = zdot
        f[:,3] = rolldot
        f[:,4] = pitchdot
        f[:,5] = yawdot
        f[:,6] = thrust*(np.cos(roll)*np.sin(pitch)*np.cos(yaw)+np.sin(roll)*np.sin(pitch))
        f[:,7] = thrust*(np.cos(roll)*np.sin(pitch)*np.sin(yaw)-np.sin(roll)*np.cos(pitch))
        f[:,8] = thrust*np.cos(roll)*np.cos(pitch)-self.g
        f[:,9] = Mx
        f[:,10] = My
        f[:,11] = Mz

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f
