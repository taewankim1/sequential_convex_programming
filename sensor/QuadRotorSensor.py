import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
from sensor import Sensor
import time

class lidar(Sensor) :
    def __init__(self,name,ix,iu) :
        super().__init__(name,ix,iu)
        self.dtheta = np.pi/16
        self.num_sensor = int(np.pi/self.dtheta+1)
        self.theta_sensor = [-i*self.dtheta for i in range(self.num_sensor)]
        self.N_theta = len(self.theta_sensor)
        self.N_lidar = 200
        self.length_lidar = 3
        self.d_lidar = self.length_lidar / self.N_lidar

    def check_obstacle(self,xt,yt,c,H) :
            x = np.array([xt,yt,0])
            for c1,H1 in zip(c,H) :
                if 1-np.linalg.norm(H1@(x-c1)) >= 0 :
                    return True
            return False

    def check_obstacle_new(self,p_mat,c,H) :
        N_data,_ = np.shape(p_mat)
        p_mat = np.expand_dims(p_mat,2)
        flag_obstacle = np.zeros(N_data) # if obstacle, True
        for c1,H1 in zip(c,H) :
            H_mat = np.repeat(np.expand_dims(H1,0),N_data,0)
            c_mat = np.expand_dims(np.repeat(np.expand_dims(c1,0),N_data,0),2)
            flag_obstacle_e = 1 - np.linalg.norm(np.squeeze(H_mat@(p_mat-c_mat)),axis=1) >=0 
            flag_obstacle = np.logical_or(flag_obstacle,flag_obstacle_e)
        return flag_obstacle

    def state2obs(self,x,c,H,method=2) :

        xdim = np.ndim(x)
        if xdim == 1: # 1 step state & input
            N = 1
            x = np.expand_dims(x,axis=0)
        else :
            N = np.size(x,axis = 0)

        observ = {}
        length_list = []
        point_list = []
        start = time.time()
        for rx in x :
            obs = []
            point = []
            # method 2
            if method == 2 :
                point_mat = np.zeros((self.N_lidar,self.N_theta,3))
                for idx_point in range(1,self.N_lidar+1) :
                    r = round(idx_point * self.d_lidar,4)
                    point_mat[idx_point-1,:,0] = rx[0] + r * np.cos(self.theta_sensor)
                    point_mat[idx_point-1,:,1] = rx[1] + r * np.sin(self.theta_sensor)
                v,h,d = np.shape(point_mat)
                point_mat = np.reshape(point_mat,(v*h,d))
                flag_obstacle = self.check_obstacle_new(point_mat,c,H)
                point_mat = np.reshape(point_mat,(v,h,d))
                flag_obstacle = np.reshape(flag_obstacle,(v,h,1))
                for idx_sensor,theta in enumerate(self.theta_sensor) :
                    for idx_point in range(1,self.N_lidar+1) :
                        r = round(idx_point * self.d_lidar,4)
                        xt = point_mat[idx_point-1,idx_sensor,0]
                        yt = point_mat[idx_point-1,idx_sensor,1]
                        if c is None :
                            flag_obs = False
                        else :
                            flag_obs = flag_obstacle[idx_point-1,idx_sensor,0]
                        if flag_obs == True :
                            break
                    obs.append(r)
                    point.append([xt,yt])
                length_list.append(obs)
                point_list.append(point)

            # method 1
            if method == 1 :
                for idx_sensor,theta in enumerate(self.theta_sensor) :
                    for idx_point in range(1,self.N_lidar+1) :
                        r = round(idx_point * self.d_lidar,4)
                        xt = rx[0] + r * np.cos(theta)
                        yt = rx[1] + r * np.sin(theta)
                        if c is None :
                            flag_obs = False
                        else :
                            flag_obs = self.check_obstacle(xt,yt,c,H)
                        if flag_obs == True :
                            break
                    obs.append(r)
                    point.append([xt,yt])

                length_list.append(obs)
                point_list.append(point)

        observ['length'] = np.array(length_list).squeeze()
        observ['point'] = np.array(point_list).squeeze()
        end = time.time()
        print(end - start)

        return observ





