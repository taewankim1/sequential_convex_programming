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
from utils import quaternion_to_euler
class Landing3D(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.m_dry = 1.0
        self.T_min = 1.5
        self.T_max = 6.5
        self.delta_max = np.deg2rad(30) # gimbal angle
        self.theta_max = np.deg2rad(90) # tilt angle
        self.gamma_gs = np.deg2rad(20)
        self.w_max = np.deg2rad(60)
        self.idx_bc_f = slice(1, 14)
        
    def forward(self,x,u,xbar,ubar,final=False):
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
        # if final == False :
        h.append(cvx.norm(x[1:3]) <= x[3] / np.tan(self.gamma_gs))
        h.append(cvx.norm(x[8:10]) <= np.sqrt((1-np.cos(self.theta_max))/2))
        h.append(cvx.norm(x[11:14]) <= self.w_max)

        # input constraints
        h.append(cvx.norm(u) <= self.T_max)
        h.append(self.T_min - ubar.T@u / cvx.norm(ubar) <= 0)
        h.append(np.cos(self.delta_max) * cvx.norm(u) <= u[2])
        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx[self.idx_bc_f] == xf[self.idx_bc_f])

        return h

    def bc_violation(self,xnn,xf) :
        position = np.linalg.norm(xnn[1:4]-xf[1:4])
        velocity = np.linalg.norm(xnn[4:7]-xf[4:7])
        roll,pitch,yaw = quaternion_to_euler(xnn[7],xnn[8],xnn[9],xnn[10])
        roll_f,pitch_f,yaw_f = quaternion_to_euler(xf[7],xf[8],xf[9],xf[10])
        attitude = np.rad2deg(np.linalg.norm(np.array([roll,pitch,yaw])-np.array([roll_f,pitch_f,yaw_f])))
        angular_rate =np.rad2deg(np.linalg.norm(xnn[11:]-xf[11:]) )
        return position,velocity,attitude,angular_rate

    def forward_full(self,x,u):
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
        m = x[:,0]
        rx = x[:,1]
        ry = x[:,2]
        rz = x[:,3]
        vx = x[:,4]
        vy = x[:,5]
        vz = x[:,6]
        q = x[:,7:11]
        w = x[:,11:14]

        q0 = x[:,7]
        q1 = x[:,8]
        q2 = x[:,9]
        q3 = x[:,10]
        
        w = x[:,11:14]
        wx = x[:,11]
        wy = x[:,12]
        wz = x[:,13]

        ux = u[:,0]
        uy = u[:,1]
        uz = u[:,2]

        h = np.zeros((N,7))
        h[:,0] = (self.m_dry - m)
        h[:,1] = np.rad2deg(np.linalg.norm(w,axis=1) - self.w_max)

        # h[:,2] = np.tan(self.gamma_gs) * np.linalg.norm(x[:,1:3],axis=1) - x[:,3]
        h[:,2] = np.rad2deg(self.gamma_gs - np.arctan(x[:,3] / np.linalg.norm(x[:,1:3],axis=1)))

        # h[:,3] = np.cos(self.theta_max) - 1 + 2*(q1**2 + q2**2)
        h[:,3] = np.rad2deg(np.arccos( 1 - 2*(q1**2 + q2**2)) - self.theta_max)

        h[:,4] = (np.linalg.norm(u,axis=1) - self.T_max)

        h[:,5] = (self.T_min - np.linalg.norm(u,axis=1))

        # h[:,6] = np.cos(self.delta_max) * np.linalg.norm(u,axis=1) - uz
        h[:,6] = np.rad2deg(np.arccos(uz / np.linalg.norm(u,axis=1)) - self.delta_max)
        return h
    
    def normalized_constraint_violation(self,x,u):
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
        m = x[:,0]
        rx = x[:,1]
        ry = x[:,2]
        rz = x[:,3]
        vx = x[:,4]
        vy = x[:,5]
        vz = x[:,6]
        q = x[:,7:11]
        w = x[:,11:14]

        q0 = x[:,7]
        q1 = x[:,8]
        q2 = x[:,9]
        q3 = x[:,10]
        
        w = x[:,11:14]
        wx = x[:,11]
        wy = x[:,12]
        wz = x[:,13]

        ux = u[:,0]
        uy = u[:,1]
        uz = u[:,2]

        h = np.zeros((N,7))
        h[:,0] = (self.m_dry - m)/self.m_dry
        h[:,1] = (np.linalg.norm(w,axis=1) - self.w_max)/self.w_max

        # h[:,2] = np.tan(self.gamma_gs) * np.linalg.norm(x[:,1:3],axis=1) - x[:,3]
        h[:,2] = (self.gamma_gs - np.arctan(x[:,3] / np.linalg.norm(x[:,1:3],axis=1)))/self.gamma_gs

        # h[:,3] = np.cos(self.theta_max) - 1 + 2*(q1**2 + q2**2)
        h[:,3] = (np.arccos( 1 - 2*(q1**2 + q2**2)) - self.theta_max) / self.theta_max

        h[:,4] = (np.linalg.norm(u,axis=1) - self.T_max) / self.T_max

        h[:,5] = (self.T_min - np.linalg.norm(u,axis=1)) / self.T_min

        # h[:,6] = np.cos(self.delta_max) * np.linalg.norm(u,axis=1) - uz
        h[:,6] = (np.arccos(uz / np.linalg.norm(u,axis=1)) - self.delta_max) / self.delta_max
        return h

    
