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
        self.idx_bc_f = slice(0, ix)
        
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
        h.append(t-np.deg2rad(90) <= 0)
        h.append(-t-np.deg2rad(90) <= 0)
        # h.append(w-np.deg2rad(60) <= 0)
        # h.append(-w-np.deg2rad(60) <= 0)
        h.append(thrust-10 <= 0)
        h.append(-thrust+0 <= 0)
        h.append(gimbal-np.deg2rad(90) <= 0)
        h.append(-gimbal-np.deg2rad(90) <= 0)
        h.append(-ry <= 0)
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h
    

    
