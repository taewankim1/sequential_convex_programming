import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
from sensor import Sensor

class quadrotorpm(Sensor) :
    def __init__(self,name,ix,iu) :
        super().__init__(name,ix,iu)
        self.dtheta = np.pi/16
        self.num_sensor = int(np.pi/self.dtheta+1)
        self.theta_sensor = [-i*self.dtheta for i in range(self.num_sensor)]
        self.N_lidar = 200
        self.length_lidar = 2
        self.d_lidar = self.length_lidar / self.N_lidar

    def check_obstacle(self,xt,yt,c,H) :
        x = np.array([xt,yt,0])
        for c1,H1 in zip(c,H) :
            if 1-np.linalg.norm(H1@(x-c1)) >= 0 :
                return True
        return False

    def state2obs(self,x,c,H) :
        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)

        observ = {}
        length_list = []
        point_list = []
        for rx in x :
            obs = []
            point = []
            for idx_sensor,theta in enumerate(self.theta_sensor) :
                for idx_point in range(self.N_lidar+1) :
                    r = round(idx_point * self.d_lidar,4)
                    xt = rx[0] + r * np.cos(theta)
                    yt = rx[1] + r * np.sin(theta)
                    flag_obs = self.check_obstacle(xt,yt,c,H)
                    if flag_obs == True :
                        break
                obs.append(r)
                point.append([xt,yt])
            length_list.append(obs)
            point_list.append(point)
        observ['length'] = np.array(length_list)
        observ['point'] = np.array(point_list)

        return observ





