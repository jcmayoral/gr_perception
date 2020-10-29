#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray, RiskIndexes, RiskObject
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty

import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import copy
from tf.transformations import euler_from_quaternion

plt.ion()
#fig, ax = plt.subplots(1,1,subplot_kw=dict(polar=True))
fig = plt.figure()
#ax = fig.add_subplot(111, projection='polar')
ax = fig.add_axes([0.1,0.1,0.8,0.8])
#global main_msg

main_msg = FoundObjectsArray()
indexes = Twist()
trajectories_x = list()
trajectories_y = list()

trajectory_id = 0


def callback(msg):
    global main_msg
    main_msg = msg
    global trajectories_x
    global trajectories_y
    #assume one person for now
    p1 = msg.objects[0].pose
    trajectories_x.append(p1.position.x)
    trajectories_y.append(p1.position.y)

def callback_indexes(msg):
    global indexes
    indexes = msg

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


cmap = color=cm.rainbow(np.linspace(0,1,20))
#get_cmap(20,'binary')
cmapdyn = get_cmap(20,'hsv')

def new_trajectory(msg):
    global trajectories_x
    global trajectories_y
    global trajectory_id

    fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.cla()
    global main_msg, indexes
    global cmap, cmapdyn
    trajectories

    narea=80
    ncount1 = 5

    ax.plot(x, y,c=cmap[trajectory_id], alpha=0.75)

    trajectories_x = list()
    trajectories_y = list()

    trajectory_id = trajectory_id + 1
    fig.canvas.draw()

if __name__ == '__main__':
    rospy.init_node("persons_status_plotter")
    subindexes = rospy.Subscriber("/next_trajectory", Empty,
                            new_trajectory, queue_size =1)
    sub = rospy.Subscriber("/pointcloud_lidar_processing/found_object", FoundObjectsArray,
                            callback, queue_size =1)
    while not rospy.is_shutdown():
        pass
        #plot()
    #rospy.spin()
    print "out"
    """
    plt.rc('font', size=20)
    plt.rc('axes', titlesize=40)
    plt.rc('axes', labelsize=20)
    #rect = patches.Rectangle((-1,-0.75),1.8,1.5)
    #ax.add_patch(rect)
    warnrect = np.array([[5.0,-5.0],[5.0,5.0],[-5.0,5.0],[-5.0,-5.0],[5.0,-5.0]])
    plt.plot(warnrect[:,0], warnrect[:,1],linewidth=4,c='y',label="Warning Zone")
    dangerrect = np.array([[0,-4.0],[4.0,-2.0],[4.0,2.0],[0,4.0],[-4.0,2.0],[-4.0,-2.0],[0.,-4.0]])
    plt.plot(dangerrect[:,0], dangerrect[:,1],linewidth=4,c='r',label="Danger Zone")
    lethalrect = np.array([[3.5,1.0],[3.5,-1],[-2.0,-1],[-2.0,1],[3.5,1.0]])
    plt.plot(lethalrect[:,0], lethalrect[:,1],linewidth=4,c='k', label="Lethal Zone")

    ax.set_xticks(np.arange(-8, 8, 1))
    ax.set_yticks(np.arange(-6, 6, 1))
    plt.xticks(fontsize=20, rotation=0)
    plt.yticks(fontsize=20, rotation=0)
    #plt.colorbar()#cmap, ax)
    """

    plt.grid(linestyle='-', linewidth=2)
    plt.xlabel("Coordinate X [m]", fontsize=30)
    plt.ylabel("Coordinate Y [m]", fontsize=30)
    plt.title("Persons Trajectories on Odometry Map")
    #  plt.legend()
    plt.savefig("trajectories_map.png")
    print "END"
    #rospy.sleep(10)
    #ax.show()
