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
    def __init__(self,name,ix,iu,delT,linearization="numeric_central"):
        super().__init__(name,ix,iu,delT,linearization)
        # self.m_wet = 2
        # self.m_dry = 0.75
        self.J = 1e-2
        self.J_mat = self.J * np.array([1,1,1])
        self.r_t = 1e-2
        self.r_t_mat = self.r_t * np.array([-1,0,0])
        self.g = 1
        self.g_mat = self.g * np.array([-1,0,0])
        self.alpha_m = 0.01

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

        q0 = x[:,7]
        q1 = x[:,8]
        q2 = x[:,9]
        q3 = x[:,10]

        wx = x[:,11]
        wy = x[:,12]
        wz = x[:,13]

        ux = u[:,0]
        uy = u[:,1]
        uz = u[:,2]

        # direction cosine matrix 
        C_B_I = self.get_dcm(q)
        C_I_B = np.transpose(C_B_I,(0, 2, 1))

        omega_w = self.get_omega(w)
        skew_w = self.get_skew(w)

        r_t_repeat = np.repeat(np.expand_dims(self.r_t_mat,0),N,axis=0) 
        skew_r_t = self.get_skew(r_t_repeat)

        J_inv = np.repeat(1/np.expand_dims(self.J_mat,0),N,axis=0)
        # output
        f = np.zeros_like(x)
        f[:,0] = -self.alpha_m*np.sqrt(ux**2 + uy**2 + uz**2)
        f[:,1] = vx
        f[:,2] = vy
        f[:,3] = vz
        f[:,4] = -self.g + ux*(-2*q2**2 - 2*q3**2 + 1)/m + uy*(-2*q0*q3 + 2*q1*q2)/m + uz*(2*q0*q2 + 2*q1*q3)/m
        f[:,5] = ux*(2*q0*q3 + 2*q1*q2)/m + uy*(-2*q1**2 - 2*q3**2 + 1)/m + uz*(-2*q0*q1 + 2*q2*q3)/m
        f[:,6] = ux*(-2*q0*q2 + 2*q1*q3)/m + uy*(2*q0*q1 + 2*q2*q3)/m + uz*(-2*q1**2 - 2*q2**2 + 1)/m
        f[:,7] = -0.5*q1*wx - 0.5*q2*wy - 0.5*q3*wz
        f[:,8] = 0.5*q0*wx + 0.5*q2*wz - 0.5*q3*wy
        f[:,9] = 0.5*q0*wy - 0.5*q1*wz + 0.5*q3*wx
        f[:,10] = 0.5*q0*wz + 0.5*q1*wy - 0.5*q2*wx
        f[:,12] = 1.0*self.r_t*uz/self.J
        f[:,13] = -1.0*self.r_t*uy/self.J

        if discrete is True :
            return np.squeeze(x + f * self.delT)
        else :
            return f

    def diff(self,x,u,discrete=True) :
        assert discrete == False
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
            u = np.expand_dims(u,axis=0)
        else :
            N = np.size(x,axis = 0)
        m = x[:,0]
        rx = x[:,1]
        ry = x[:,2]
        rz = x[:,3]
        vx = x[:,4]
        vy = x[:,5]
        vz = x[:,6]

        q0 = x[:,7]
        q1 = x[:,8]
        q2 = x[:,9]
        q3 = x[:,10]

        wx = x[:,11]
        wy = x[:,12]
        wz = x[:,13]
        
        ux = u[:,0]
        uy = u[:,1]
        uz = u[:,2]

        fx = np.zeros((N,ix,ix))
        fx[:,1,4] = 1
        fx[:,2,5] = 1
        fx[:,3,6] = 1
        fx[:,4,0] = -ux*(-2*q2**2 - 2*q3**2 + 1)/m**2 - uy*(-2*q0*q3 + 2*q1*q2)/m**2 - uz*(2*q0*q2 + 2*q1*q3)/m**2
        fx[:,4,7] = 2*q2*uz/m - 2*q3*uy/m
        fx[:,4,8] = 2*q2*uy/m + 2*q3*uz/m
        fx[:,4,9] = 2*q0*uz/m + 2*q1*uy/m - 4*q2*ux/m
        fx[:,4,10] = -2*q0*uy/m + 2*q1*uz/m - 4*q3*ux/m
        fx[:,5,0] = -ux*(2*q0*q3 + 2*q1*q2)/m**2 - uy*(-2*q1**2 - 2*q3**2 + 1)/m**2 - uz*(-2*q0*q1 + 2*q2*q3)/m**2
        fx[:,5,7] = -2*q1*uz/m + 2*q3*ux/m
        fx[:,5,8] = -2*q0*uz/m - 4*q1*uy/m + 2*q2*ux/m
        fx[:,5,9] = 2*q1*ux/m + 2*q3*uz/m
        fx[:,5,10] = 2*q0*ux/m + 2*q2*uz/m - 4*q3*uy/m
        fx[:,6,0] = -ux*(-2*q0*q2 + 2*q1*q3)/m**2 - uy*(2*q0*q1 + 2*q2*q3)/m**2 - uz*(-2*q1**2 - 2*q2**2 + 1)/m**2
        fx[:,6,7] = 2*q1*uy/m - 2*q2*ux/m
        fx[:,6,8] = 2*q0*uy/m - 4*q1*uz/m + 2*q3*ux/m
        fx[:,6,9] = -2*q0*ux/m - 4*q2*uz/m + 2*q3*uy/m
        fx[:,6,10] = 2*q1*ux/m + 2*q2*uy/m
        fx[:,7,8] = -0.5*wx
        fx[:,7,9] = -0.5*wy
        fx[:,7,10] = -0.5*wz
        fx[:,7,11] = -0.5*q1
        fx[:,7,12] = -0.5*q2
        fx[:,7,13] = -0.5*q3
        fx[:,8,7] = 0.5*wx
        fx[:,8,9] = 0.5*wz
        fx[:,8,10] = -0.5*wy
        fx[:,8,11] = 0.5*q0
        fx[:,8,12] = -0.5*q3
        fx[:,8,13] = 0.5*q2
        fx[:,9,7] = 0.5*wy
        fx[:,9,8] = -0.5*wz
        fx[:,9,10] = 0.5*wx
        fx[:,9,11] = 0.5*q3
        fx[:,9,12] = 0.5*q0
        fx[:,9,13] = -0.5*q1
        fx[:,10,7] = 0.5*wz
        fx[:,10,8] = 0.5*wy
        fx[:,10,9] = -0.5*wx
        fx[:,10,11] = -0.5*q2
        fx[:,10,12] = 0.5*q1
        fx[:,10,13] = 0.5*q0

        alpha_m = self.alpha_m
        J = self.J
        r_t = self.r_t

        fu = np.zeros((N,ix,iu))
        
        fu[:,0,0] = -alpha_m*ux/np.sqrt(ux**2 + uy**2 + uz**2)
        fu[:,0,1] = -alpha_m*uy/np.sqrt(ux**2 + uy**2 + uz**2)
        fu[:,0,2] = -alpha_m*uz/np.sqrt(ux**2 + uy**2 + uz**2)
        fu[:,4,0] = (-2*q2**2 - 2*q3**2 + 1)/m
        fu[:,4,1] = (-2*q0*q3 + 2*q1*q2)/m
        fu[:,4,2] = (2*q0*q2 + 2*q1*q3)/m
        fu[:,5,0] = (2*q0*q3 + 2*q1*q2)/m
        fu[:,5,1] = (-2*q1**2 - 2*q3**2 + 1)/m
        fu[:,5,2] = (-2*q0*q1 + 2*q2*q3)/m
        fu[:,6,0] = (-2*q0*q2 + 2*q1*q3)/m
        fu[:,6,1] = (2*q0*q1 + 2*q2*q3)/m
        fu[:,6,2] = (-2*q1**2 - 2*q2**2 + 1)/m
        fu[:,12,2] = 1.0*r_t/J
        fu[:,13,1] = -1.0*r_t/J

        return fx.squeeze(),fu.squeeze()

