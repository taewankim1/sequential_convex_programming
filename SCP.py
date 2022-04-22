from scipy.integrate import odeint
from scipy.integrate import solve_ivp
import numpy as np
import cvxpy as cvx
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
    # print ("Values are: \n%s" % (x))

import cost
import model

from Scaling import TrajectoryScaling

class SCP:
    def __init__(self,name) :
        self.name = name

    def initialize(self) :
        return