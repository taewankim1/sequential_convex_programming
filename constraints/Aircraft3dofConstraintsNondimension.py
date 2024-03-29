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

class Aircraft3dofNondimension(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.CL_min = -0.31
        self.CL_max = 1.52
        self.phi_min = -np.deg2rad(15)
        self.phi_max = np.deg2rad(15)
        self.T_min = 0
        self.T_max = 1126.3 * 1e3

        self.v_min = 95
        self.v_max = 270

        self.gamma_min = -np.deg2rad(30)
        self.gamma_max = np.deg2rad(30) * 0

        self.ih = 11

    def set_scale(self,scl_v,scl_f) :
        self.scl_v = scl_v
        self.scl_f = scl_f
        
    def forward(self,x,u,xbar=None,ubar=None,final=False):
        # state & input
        rx = x[0]
        ry = x[1]
        rz = x[2]
        v = x[3] # speed
        gamma = x[4] # path angle
        psi = x[5] # velocity heading
        
        CL = u[0] # lift coefficient
        phi = u[1] # bank angle
        thrust = u[2] # thrust

        # # scale
        T_min = self.T_min / self.scl_f
        T_max = self.T_max / self.scl_f
        v_max = self.v_max / self.scl_v
        v_min = self.v_min / self.scl_v

        h = []
        h.append(CL>=self.CL_min)
        h.append(CL<=self.CL_max)
        h.append(phi>=self.phi_min)
        h.append(phi<=self.phi_max)
        h.append(thrust>=T_min)
        h.append(thrust<=T_max)
        h.append(v>=v_min)
        h.append(v<=v_max)
        h.append(gamma>=self.gamma_min)
        h.append(gamma<=self.gamma_max)
        h.append(rz>=0)
        return h

    def forward_buffer(self,x,u,bf):
        # state & input
        rx = x[0]
        ry = x[1]
        rz = x[2]
        v = x[3] # speed
        gamma = x[4] # path angle
        psi = x[5] # velocity heading
        
        CL = u[0] # lift coefficient
        phi = u[1] # bank angle
        thrust = u[2] # thrust

        h = []
        h.append(CL>=bf[0] + self.CL_min)
        h.append(CL+bf[1]<=self.CL_max)
        h.append(phi>=bf[2]+self.phi_min)
        h.append(phi+bf[3]<=self.phi_max)
        h.append(thrust>=bf[4]+self.T_min)
        h.append(thrust+bf[5]<=self.T_max)
        h.append(v>=bf[6]+self.v_min)
        h.append(v+bf[7]<=self.v_max)
        h.append(gamma>=bf[8]+self.gamma_min)
        h.append(gamma+bf[9]<=self.gamma_max)
        h.append(rz >= bf[10])
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h

class Aircraft3dofStateTriggered(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.CL_min = -0.31
        self.CL_max = 1.52
        self.phi_min = -np.deg2rad(15)
        self.phi_max = np.deg2rad(15)
        self.T_min = 0
        self.T_max = 1#1126.3 * 1e3
        self.v_min = 95
        self.v_max = 270
        self.gamma_min = -np.deg2rad(30)
        self.gamma_max = np.deg2rad(30) * 0
        self.scl_v = 1
        self.scl_f = 1
        self.ih = 11

        self.z_trigger = 2000
        self.y_min_trigger = -65000
        self.y_max_trigger = -55000

    def set_scale(self,scl_v,scl_f) :
        self.scl_v = scl_v
        self.scl_f = scl_f
        
    def forward(self,x,u,xbar=None,ubar=None,final=False):
        # state & input
        rx = x[0]
        ry = x[1]
        rz = x[2]
        v = x[3] # speed
        gamma = x[4] # path angle
        psi = x[5] # velocity heading
        
        CL = u[0] # lift coefficient
        phi = u[1] # bank angle
        thrust = u[2] # thrust

        # # scale
        T_min = self.T_min / self.scl_f
        T_max = self.T_max / self.scl_f
        v_max = self.v_max / self.scl_v
        v_min = self.v_min / self.scl_v

        h = []
        h.append(CL>=self.CL_min)
        h.append(CL<=self.CL_max)
        h.append(phi>=self.phi_min)
        h.append(phi<=self.phi_max)
        h.append(thrust>=T_min)
        h.append(thrust<=T_max)
        h.append(v>=v_min)
        h.append(v<=v_max)
        h.append(gamma>=self.gamma_min)
        h.append(gamma<=self.gamma_max)
        h.append(rz>=0)

        # state-triggered
        rybar = xbar[1]
        rzbar = xbar[2]
        if rzbar < self.z_trigger :
            h1bar = -(rzbar-self.z_trigger) * (rybar-self.y_max_trigger) 
            rh1_rz = -(rybar-self.y_max_trigger) * (rz-rzbar)
            rh1_ry = -(rzbar-self.z_trigger) * (ry-rybar)
            h.append(h1bar+rh1_rz+rh1_ry <= 0)

            h2bar = -(rzbar-self.z_trigger) * (-rybar+self.y_min_trigger) 
            rh2_rz = (rybar-self.y_min_trigger) * (rz-rzbar)
            rh2_ry = (rzbar-self.z_trigger) * (ry-rybar)
            h.append(h2bar+rh2_rz+rh2_ry <= 0)
        return h


    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h



class Aircraft3dofApprox(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.CL_min = -0.31
        self.CL_max = 1.52
        self.phi_min = -np.deg2rad(15)
        self.phi_max = np.deg2rad(15)
        self.T_min = 0
        self.T_max = 1#1126.3 * 1e3
        self.v_min = 95
        self.v_max = 270
        self.gamma_min = -np.deg2rad(20)
        self.gamma_max = np.deg2rad(20)*0
        self.scl_v = 1
        self.scl_f = 1
        self.ih = 11

    def set_scale(self,scl_v,scl_f) :
        self.scl_v = scl_v
        self.scl_f = scl_f
        
    def forward(self,x,u,xbar=None,ubar=None,final=False):
        # state & input
        rx = x[0]
        ry = x[1]
        rz = x[2]
        v = x[3] # speed
        gamma = x[4] # path angle
        psi = x[5] # velocity heading
        
        alpha = u[0] # lift coefficient
        phi = u[1] # bank angle
        thrust = u[2] # thrust

        CLalpha = 4.2
        CL0 = 0.4225
        CL = CL0 + CLalpha * alpha

        # # scale
        T_min = self.T_min / self.scl_f
        T_max = self.T_max / self.scl_f
        v_max = self.v_max / self.scl_v
        v_min = self.v_min / self.scl_v

        h = []
        h.append(CL>=self.CL_min)
        h.append(CL<=self.CL_max)
        h.append(phi>=self.phi_min)
        h.append(phi<=self.phi_max)
        h.append(thrust>=T_min)
        h.append(thrust<=T_max)
        h.append(v>=v_min)
        h.append(v<=v_max)
        h.append(gamma>=self.gamma_min)
        h.append(gamma<=self.gamma_max)
        h.append(rz>=0)
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h
    

    
