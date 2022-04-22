import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

from model import OptimalcontrolModel



class Entry3dofSpherical(OptimalcontrolModel):
    def __init__(self,name,ix,iu,linearization="numeric_central"):
        super().__init__(name,ix,iu,linearization)
        self.ge = 9.81
        self.re = 6371*1e3
        self.rhoe = 1.3
        self.H = 7e3
        self.beta = 1/self.H

        self.nt = np.sqrt(self.re/self.ge)
        self.nd = self.re
        self.nv = np.sqrt(self.ge*self.re)

        self.Kq = 1.2035 * 1e-5
        self.mass = 104305
        self.Sref = 391.22
        self.alpha_param = np.deg2rad(40)
        self.alpha_deg = np.rad2deg(self.alpha_param)
        self.B = self.re * self.Sref / (2*self.mass)
        
    def forward(self,x,u,idx=None):
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 0 :
            u = np.array([u])
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)

        # state & input
        r = x[:,0]
        theta = x[:,1]
        phi = x[:,2]
        v = x[:,3] # speed
        gamma = x[:,4] # path angle
        psi = x[:,5] # velocity heading

        sigma = u[:,0] # lift coefficient

        nv,rhoe,beta,re,B = self.nv,self.rhoe,self.beta,self.re,self.B

        # extrfxt sines and cosines of various values    
        cp  = np.cos(phi)
        sp  = np.sin(phi)
        cg  = np.cos(gamma)
        sg  = np.sin(gamma)
        cps = np.cos(psi)
        sps = np.sin(psi)
        cs  = np.cos(sigma)
        ss  = np.sin(sigma)

        # Determine lift and drag coefficients from velocity
        flag_tmp = nv*v > 4750
        alpha = 40-0.20705*((v*nv-4570)**2)/(340**2)
        alpha[flag_tmp] = 40

        Cl          = -0.041065+0.016292*alpha+0.0002602*alpha**2
        Cd          = 0.080505-0.03026*Cl+0.86495*Cl**2

        # determine atm dens, lift, drag quantities
        rho     = rhoe*np.exp(-beta*re*(r - 1 ))
        L       = B*rho*Cl*v**2
        D       = B*rho*Cd*v**2

        # output
        f = np.zeros_like(x)

        f[:,0] = v*sg
        f[:,1] = v*cg*sps/(r*cp)
        f[:,2] = v*cg*cps/r
        f[:,3] = -D-sg/(r**2)
        f[:,4] = (1/v)*(L*cs+(v**2-1/r)*cg/r)
        f[:,5] = (1/v)*(L*ss/cg+(v**2)*cg*sps*np.tan(phi)/r)

        return f

    def diff(self,x,u) :
        # state & input size
        ix = self.ix
        iu = self.iu
        
        ndim = np.ndim(x)
        if ndim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)
        udim = np.ndim(u)
        if udim == 0 :
            u = np.array([u])
        udim = np.ndim(u)
        if udim == 1 :
            u = np.expand_dims(u,axis=0)

        # state & input
        r = x[:,0]
        theta = x[:,1]
        phi = x[:,2]
        v = x[:,3] # speed
        gamma = x[:,4] # path angle
        psi = x[:,5] # velocity heading

        sigma = u[:,0] # lift coefficient

        nv,rhoe,beta,re,B = self.nv,self.rhoe,self.beta,self.re,self.B

        # extrfxt sines and cosines of various values    
        cp  = np.cos(phi)
        sp  = np.sin(phi)
        cg  = np.cos(gamma)
        sg  = np.sin(gamma)
        cps = np.cos(psi)
        sps = np.sin(psi)
        cs  = np.cos(sigma)
        ss  = np.sin(sigma)

        # Determine lift and drag coefficients from velocity
        flag_tmp = nv*v > 4750
        alpha = 40-0.20705*((v*nv-4570)**2)/(340**2)
        alpha[flag_tmp] = 40

        Cl          = -0.041065+0.016292*alpha+0.0002602*alpha**2
        Cd          = 0.080505-0.03026*Cl+0.86495*Cl**2

        # determine atm dens, lift, drag quantities
        rho     = rhoe*np.exp(-beta*re*(r - 1 ))
        L       = B*rho*Cl*v**2
        D       = B*rho*Cd*v**2
        # determine derivatives of above quanitites
        drho    = -beta*re*rho
        Lr      = B*Cl*(v**2)*drho
        Lv      = 2*B*Cl*rho*v
        Dr      = B*Cd*(v**2)*drho
        Dv      = 2*B*Cd*rho*v


        fx = np.zeros((N,ix,ix))
        fx[:,0,3]  = sg
        fx[:,0,4]  = v*cg
        fx[:,1,0]  = -v*cg*sps/((r**2)*cp)
        fx[:,1,2]  = v*cg*sps*sp/(r*(cp**2))
        fx[:,1,3]  = cg*sps/(r*cp)
        fx[:,1,4]  = -v*sg*sps/(r*cp)
        fx[:,1,5]  = v*cg*cps/(r*cp)
        fx[:,2,0]  = -v*cg*cps/(r**2)
        fx[:,2,3]  = cg*cps/r
        fx[:,2,4]  = -v*sg*cps/r
        fx[:,2,5]  = -v*cg*sps/r
        fx[:,3,0]  = -Dr + 2*sg/(r**3)
        fx[:,3,3]  = -Dv
        fx[:,3,4]  = -cg/(r**2)
        fx[:,4,0]  = (cs/v)*Lr-v*cg/(r**2)+2*cg/(v*(r**3))
        fx[:,4,3]  = (cs/v)*(Lv-L/v)+cg/r+cg/((v**2)*(r**2))
        fx[:,4,4]  = (1/(v*r) -v)*sg/r
        fx[:,5,0]  = ss*Lr/(v*cg)-v*cg*sps*np.tan(phi)/(r**2)
        fx[:,5,2]  = v*cg*sps/(r*(cp**2))
        fx[:,5,3]  = (1/v)*(ss*Lv/cg)+ cg*sps*np.tan(phi)/r - L*ss/(cg*(v**2))
        fx[:,5,4]  = L*ss*sg/((cg**2)*v) -(v/r)*sg*sps*np.tan(phi)
        fx[:,5,5]  = (v/r)*cg*cps*np.tan(phi)

        fu = np.zeros((N,ix,iu))
        
        fu[:,4,0] = -L*ss/v; 
        fu[:,5,0] = L*cs/(v*cg); 

        return fx,fu