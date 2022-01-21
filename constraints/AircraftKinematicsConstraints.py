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

class AircraftKinematics(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.CL_min = -0.27
        self.CL_max = 1.73
        self.phi_min = -np.deg2rad(30)
        self.phi_max = np.deg2rad(30)
        self.T_min = -1126.3*1e3
        self.T_max = 1126.3 * 1e3
        self.v_min = 0
        self.v_max = 270
        self.gamma_min = -np.deg2rad(20)
        self.gamma_max = 0
        
    def forward(self,x,u,xbar=None,ubar=None,final=False):
        # state & input
        rx = x[0]
        ry = x[1]
        rz = x[2]
        v = x[3] # speed
        gamma = x[4] # path angle
        psi = x[5] # velocity heading
        
        gamma_dot = u[0] 
        psi_dot = u[1] 
        thrust = u[2] # thrust

        h = []
        h.append(thrust>=self.T_min)
        h.append(thrust<=self.T_max)
        h.append(v>=self.v_min)
        h.append(v<=self.v_max)
        h.append(gamma>=self.gamma_min)
        h.append(gamma<=self.gamma_max)
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h
    

    
