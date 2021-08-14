import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.linalg
import time
import random
import cvxpy as cvx
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    print ("Values are: \n%s" % (x))

from constraints import OptimalcontrolConstraints
import IPython

class Landing3D(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,ih):
        super().__init__(name,ix,iu,ih)
        self.m_dry = 0.75
        self.T_min = 0.3
        self.T_max = 3.0
        self.delta_max = np.deg2rad(20) # gimbal angle
        self.theta_max = np.deg2rad(30) # tilt angle
        self.gamma_gs = np.deg2rad(20)
        self.w_max = np.deg2rad(60)
        self.idx_bc_f = slice(1, 14)
        
    def forward(self,x,u,xbar,ubar,u_epi=0):
        # state & input
        m = x[0]
        rx = x[1]
        ry = x[2]
        rz = x[3]
        vx = x[4]
        vy = x[5]
        vz = x[6]
        q = x[7:11]
        w = x[11:14]
        
        ux = u[0]
        uy = u[1]
        uz = u[2]

        h = []
        # state constraints
        h.append(m >= self.m_dry)
        h.append(cvx.norm(x[1:3]) <= x[3] / np.tan(self.gamma_gs))
        h.append(cvx.norm(x[8:10]) <= np.sqrt((1-np.cos(self.theta_max))/2))
        h.append(cvx.norm(x[11:14]) <= self.w_max)

        # input constraints
        h.append(cvx.norm(u) <= self.T_max)
        h.append(self.T_min - np.transpose(np.expand_dims(ubar,1))@u / np.linalg.norm(ubar) <= 0)
        h.append(np.cos(self.delta_max) * cvx.norm(u) <= u[2])
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx[self.idx_bc_f] == xf[self.idx_bc_f])

        return h
    

    
