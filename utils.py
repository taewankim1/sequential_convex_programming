import imageio
import os
from mpl_toolkits.mplot3d import art3d

import matplotlib.pyplot as plt
import numpy as np
import time
import random
def print_np(x):
    print ("Type is %s" % (type(x)))
    print ("Shape is %s" % (x.shape,))
#     print ("Values are: \n%s" % (x))

# vector scaling
thrust_scale = 0.1
attitude_scale = 0.3

def make_trajectory_fig(x,u) :
    filenames = []
    N = np.shape(x)[0]
    for k in range(N):
        
        fS = 18
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel('X, east')
        ax.set_ylabel('Y, north')
        ax.set_zlabel('Z, up')
        rx, ry, rz = x[k,1:4]
        qw, qx, qy, qz = x[k,7:11]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        if k != N-1 :
            Fx, Fy, Fz = np.dot(np.transpose(CBI), u[k, :])

        # attitude vector
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')
        scale = x[0, 3]
        ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

        pad = plt.Circle((0, 0), 0.2, color='lightgrey')
        ax.add_patch(pad)
        art3d.pathpatch_2d_to_3d(pad)

        ax.plot(x[:, 1], x[:, 2], x[:, 3])
        
        filename = '../images/{:d}.png'.format(k)
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()

    with imageio.get_writer('../images/Landing3D.gif', mode='I') as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    for filename in set(filenames):
        os.remove(filename)

def plot_rocket3d(fig, x, u, xppg):
    ax = fig.add_subplot(111, projection='3d')

    N = np.shape(x)[0]

    ax.set_xlabel('X, east')
    ax.set_ylabel('Y, north')
    ax.set_zlabel('Z, up')

    for k in range(N):
        rx, ry, rz = x[k,1:4]
        qw, qx, qy, qz = x[k,7:11]

        CBI = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy + qw * qz), 2 * (qx * qz - qw * qy)],
            [2 * (qx * qy - qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz + qw * qx)],
            [2 * (qx * qz + qw * qy), 2 * (qy * qz - qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]
        ])

        dx, dy, dz = np.dot(np.transpose(CBI), np.array([0., 0., 1.]))
        if k != N-1 :
            Fx, Fy, Fz = np.dot(np.transpose(CBI), u[k, :])

        # attitude vector
        ax.quiver(rx, ry, rz, dx, dy, dz, length=attitude_scale, arrow_length_ratio=0.0, color='blue')

        # thrust vector
        ax.quiver(rx, ry, rz, -Fx, -Fy, -Fz, length=thrust_scale, arrow_length_ratio=0.0, color='red')

    scale = x[0, 3]
    ax.auto_scale_xyz([-scale / 2, scale / 2], [-scale / 2, scale / 2], [0, scale])

    pad = plt.Circle((0, 0), 0.2, color='lightgrey')
    ax.add_patch(pad)
    art3d.pathpatch_2d_to_3d(pad)

    ax.plot(x[:, 1], x[:, 2], x[:, 3])
    ax.plot(xppg[:, 1], xppg[:, 2], xppg[:, 3],'--')
#     ax.set_aspect('equal')