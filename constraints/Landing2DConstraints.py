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

class Landing2D(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,ih):
        super().__init__(name,ix,iu,ih)
        
    def forward(self,x,u,xbar=None,ubar=None):


        # state & input
        rx = x[0]
        ry = x[1]
        vx = x[2]
        vy = x[3]
        t = x[4]
        w = x[5]
        
        gimbal = u[0]
        thrust = u[1]

        h = []
        h.append(t-np.deg2rad(60) <= 0)
        h.append(-t-np.deg2rad(60) <= 0)
        h.append(w-np.deg2rad(60) <= 0)
        h.append(-w-np.deg2rad(60) <= 0)
        h.append(thrust-10 <= 0)
        h.append(-thrust-0 <= 0)
        h.append(gimbal-np.deg2rad(90) <= 0)
        h.append(-gimbal-np.deg2rad(90) <= 0)
        h.append(-ry <= 0)
        return h

        # xdim = np.ndim(x)
        # if xdim == 1: # 1 step state & input
        #     N = 1
        #     x = np.expand_dims(x,axis=0)
        # else :
        #     N = np.size(x,axis = 0)
        # udim = np.ndim(u)
        # if udim == 1 :
        #     u = np.expand_dims(u,axis=0)
     
        # # state & input
        # rx = x[:,0]
        # ry = x[:,1]
        # vx = x[:,2]
        # vy = x[:,3]
        # t = x[:,4]
        # w = x[:,5]
        
        # gimbal = u[:,0]
        # thrust = u[:,1]
        
        # # output
        # f = np.zeros((N,self.ih))
        # f[:,0] = t-np.deg2rad(60)
        # f[:,1] = -t-np.deg2rad(60)
        # f[:,2] = w-np.deg2rad(60)
        # f[:,3] = -w-np.deg2rad(60)
        # f[:,4] = thrust-3
        # f[:,5] = -thrust + 0.0 # doesn't work for the value > 0, why?
        # f[:,6] = gimbal-np.deg2rad(90)
        # f[:,7] = -gimbal-np.deg2rad(90)
        # f[:,8] = -ry  

        # return f
    

    
