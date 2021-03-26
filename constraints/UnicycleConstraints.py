import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))


from constraints import OptimalcontrolConstraints

class UnicycleConstraints(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,ih):
        super().__init__(name,ix,iu,ih)
        
    def forward(self,x,u):
        
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
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]
        
        # output
        f = np.zeros((N,self.ih))
        f[:,0] = - v - 0
        f[:,1] = -w - 10 

        return f
    
    def diff(self,x,u):

        # dimension
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        
        # state & input
        x1 = x[:,0]
        x2 = x[:,1]
        x3 = x[:,2]
        
        v = u[:,0]
        w = u[:,1]    
        
        fx = np.zeros((N,self.ih,self.ix))
        fx[:,0,0] = 0.0
        fx[:,0,1] = 0.0
        fx[:,0,2] = 0.0
        fx[:,1,0] = 0.0
        fx[:,1,1] = 0.0
        fx[:,1,2] = 0.0

        
        fu = np.zeros((N,self.ih,self.iu))
        fu[:,0,0] = 1.0
        fu[:,0,1] = 0.0
        fu[:,1,0] = 0.0
        fu[:,1,1] = 1.0
        
        return np.squeeze(fx) , np.squeeze(fu)
    
