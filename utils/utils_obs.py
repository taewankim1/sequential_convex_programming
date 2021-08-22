import imageio
import os
import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))
def generate_obstacle(point_range,dice=-1,r_safe=0.3) :
    def get_H_obs(r) :
        return np.diag([1/r,1/r,0])
    def get_H_safe(r,r_safe) :
        return np.diag([1/(r+r_safe),1/(r+r_safe),0])
    
#     dice = np.random.randint(low=1, high=4)
    c_list = []
    H_obs_list = []
    H_safe_list = []
    r_list = []
    if dice != 1 :
        c = np.array([1+np.random.uniform(-point_range,point_range),4+np.random.uniform(0,1),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 2 :
        c = np.array([1+np.random.uniform(-point_range,point_range),2+np.random.uniform(-1,0),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 3 :
        c = np.array([-1+np.random.uniform(-point_range,point_range),4+np.random.uniform(0,1),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)
    if dice != 4 :
        c = np.array([-1+np.random.uniform(-point_range,point_range),2+np.random.uniform(-1,0),0])
        r = np.random.uniform(0.2,0.5)
        c_list.append(c)
        H_obs_list.append(get_H_obs(r))
        H_safe_list.append(get_H_safe(r,r_safe))
        r_list.append(r)


    assert len(c_list) == len(H_obs_list)
    assert len(c_list) == len(H_safe_list)
    num_obstacle = len(c)
    return c_list,H_obs_list,H_safe_list,r_list