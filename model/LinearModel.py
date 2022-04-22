
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

class Linear(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT):
        super().__init__(name,ix,iu,delT)
        
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
        px = x[:,0]
        vx = x[:,1]
        py = x[:,2]
        vy = x[:,3]
        
        fx = u[:,0]
        fy = u[:,1]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = fx
        f[:,2] = vy
        f[:,3] = fy - 1

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            # print("hello")
            return f
    