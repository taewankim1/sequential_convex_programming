 coding: utf-8

# In[ ]:

import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel

class planar2D(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT):
        supear().__init__(name,ix,iu,delT)
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
        x = x[:,0]
        y = x[:,1]
        vx = x[:,2]
        vy = x[:,3]
        
        u1 = u[:,0]
        u2 = u[:,1]
        u3 = u[:,2]
        
        # output
        f = np.zeros_like(x)
        f[:,0] = vx
        f[:,1] = vy 
        f[:,2] = 

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f