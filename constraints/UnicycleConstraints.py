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
    def __init__(self,name,ix,iu):
        super().__init__(name,ix,iu)
        self.idx_bc_f = slice(0, ix)
        self.ih = 4
        
    def forward(self,x,u,xbar=None,ybar=None,idx=None):

        v = u[0]
        w = u[1]

        h = []
        h.append(v-0.2 <= 0)
        h.append(v >= -0.2)
        h.append(w<=np.deg2rad(5))
        h.append(w>=-np.deg2rad(5))

        return h

    def forward_buffer(self,x,u,bf):

        v = u[0]
        w = u[1]

        h = []
        h.append(v-0.2+bf[0] <= 0)
        h.append(v >=bf[1] -0.2)
        h.append(w+bf[2]<=np.deg2rad(5))
        h.append(w>=bf[3]-np.deg2rad(5))

        return h

    def bc_final(self,x_cvx,xf):
        h = []
        h.append(x_cvx == xf)

        return h
        


