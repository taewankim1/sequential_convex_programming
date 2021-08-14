import matplotlib.pyplot as plt
import numpy as np
import time
import random
import IPython
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel


class Landing3D(OptimalcontrolModel):
    def __init__(self,name,ix,iu,delT):
        super().__init__(name,ix,iu,delT)
        # self.m_wet = 2
        # self.m_dry = 0.75
        self.J = 1e-2 * np.array([1,1,1])
        self.r_t = 1e-2*np.array([0,0,-1])
        self.g = np.array([0,0,-1])
        self.alpha_m = 0.1

    def get_dcm(self,q) :
        N = q.shape[0]

        C_B_I = np.zeros((N,3,3))
        C_B_I[:,0,0] = 1 - 2 * (q[:,2] ** 2 + q[:,3] ** 2)
        C_B_I[:,0,1] = 2 * (q[:,1] * q[:,2] + q[:,0] * q[:,3])
        C_B_I[:,0,2] = 2 * (q[:,1] * q[:,3] - q[:,0] * q[:,2])
        C_B_I[:,1,0] = 2 * (q[:,1] * q[:,2] - q[:,0] * q[:,3])
        C_B_I[:,1,1] = 1 - 2 * (q[:,1] ** 2 + q[:,3] ** 2)
        C_B_I[:,1,2] = 2 * (q[:,2] * q[:,3] + q[:,0] * q[:,1])
        C_B_I[:,2,0] = 2 * (q[:,1] * q[:,3] + q[:,0] * q[:,2])
        C_B_I[:,2,1] = 2 * (q[:,2] * q[:,3] - q[:,0] * q[:,1])
        C_B_I[:,2,2] = 1 - 2 * (q[:,1] ** 2 + q[:,2] ** 2)

        return C_B_I

    def get_omega(self,xi) :
        N = xi.shape[0]

        xi_x = xi[:,0]
        xi_y = xi[:,1]
        xi_z = xi[:,2]

        omega = np.zeros((N,4,4))

        omega[:,0,0] = 0
        omega[:,0,1] = -xi_x
        omega[:,0,2] = -xi_y
        omega[:,0,3] = -xi_z

        omega[:,1,0] = xi_x
        omega[:,1,1] = 0
        omega[:,1,2] = xi_z
        omega[:,1,3] = -xi_y

        omega[:,2,0] = xi_y
        omega[:,2,1] = -xi_z
        omega[:,2,2] = 0
        omega[:,2,3] = xi_x
        
        omega[:,3,0] = xi_z
        omega[:,3,1] = xi_y
        omega[:,3,2] = -xi_x
        omega[:,3,3] = 0

        return omega

    def get_skew(self,xi) :
        N = xi.shape[0]

        xi_x = xi[:,0]
        xi_y = xi[:,1]
        xi_z = xi[:,2]

        skew = np.zeros((N,3,3))

        skew[:,0,0] = 0
        skew[:,0,1] = -xi_z
        skew[:,0,2] = xi_y

        skew[:,1,0] = xi_z
        skew[:,1,1] = 0
        skew[:,1,2] = -xi_x

        skew[:,2,0] = -xi_y
        skew[:,2,1] = xi_x
        skew[:,2,2] = 0

        return skew

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
        
        # state = [m rx(east) ry(north) rz(up) vx vy vz q0 q1 q2 q3 wx wy wz]
        # input = [ux uy uz]
     
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
        
        ux = u[:,0]
        uy = u[:,1]
        uz = u[:,2]

        # direction cosine matrix 
        C_B_I = self.get_dcm(q)
        C_I_B = np.transpose(C_B_I,(0, 2, 1))

        omega_w = self.get_omega(w)
        skew_w = self.get_skew(w)

        r_t_repeat = np.repeat(np.expand_dims(self.r_t,0),N,axis=0) 
        skew_r_t = self.get_skew(r_t_repeat)

        J_inv = np.repeat(1/np.expand_dims(self.J,0),N,axis=0)
        
        # output
        f = np.zeros_like(x)
        f[:,0] = - self.alpha_m * np.linalg.norm(u,axis=1)
        f[:,1:4] = x[:,4:7]
        f[:,4:7] = 1/x[:,0:1]*(C_I_B@np.expand_dims(u,2)).squeeze()+np.repeat(np.expand_dims(self.g,0),N,axis=0) 
        f[:,7:11] = 0.5*(omega_w@np.expand_dims(q,2)).squeeze()
        f[:,11:14] = J_inv * (skew_r_t@np.expand_dims(u,2)).squeeze() - \
                         (skew_w@np.expand_dims(w,2)).squeeze()

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f