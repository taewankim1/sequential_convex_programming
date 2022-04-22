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

class Linear(OptimalcontrolConstraints):
    def __init__(self,name,ix,iu,ih):
        super().__init__(name,ix,iu,ih)
        
    def forward(self,x,u):

        v = u[0]
        w = u[1]

        h = []
        # h.append(v-0.2 <= 0)
        # h.append(-w+0 <= 0)

        return h

