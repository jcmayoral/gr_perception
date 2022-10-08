#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray, RiskIndexes, RiskObject
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty, Float32
import time
import rospy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
import copy
from tf.transformations import euler_from_quaternion

plt.ion()
#fig, ax = plt.subplots(1,1,subplot_kw=dict(polar=True))
fig, axs = plt.subplots(2,1)
#fig.suptitle('ubplots')
#axs = [0,0]
#axs[0].plot(x, y)
#axs[1].plot(x, -y)
#axs[0] = fig.add_subplot(211)#, projection='polar')
#axs[1] = fig.add_subplot(212)#, projection='polar')

#ax = fig.add_axes([0.1,0.1,0.8,0.8])
#global main_msg
#main_msg = Float32()
scores = list()
speed = list()

trajectory_id = 0


def callback(msg):
    global main_msg
    global scores
    scores.append(msg.data)

def vel_callback(msg):
    global speed
    speed.append(msg.linear.x)


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


cmap = color=plt.cm.rainbow(np.linspace(0,1,22))
#get_cmap(20,'binary')
cmapdyn = get_cmap(20,'hsv')
ready = False

def new_trajectory(msg):
    global ready
    ready = True

def plot():
    global ready
    if not ready:
        #print "not finished"
        return False
    global scores
    global speed
    global trajectory_id

    #fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.cla()
    global cmap, cmapdyn

    narea=80
    ncount1 = 5

    x = np.arange(len(speed))*0.1
    axs[0].plot(x, speed,c=cmap[trajectory_id], label="safe_speed_"+str(trajectory_id), alpha=0.75, linewidth=5)
    x = np.arange(len(scores))*0.1
    axs[1].plot(x, scores,c=cmap[trajectory_id], label="braking_signal_"+str(trajectory_id), alpha=0.75, linewidth=5)


    speed = list()
    scores = list()

    trajectory_id = trajectory_id + 1
    #fig.canvas.draw()
    ready = False
    return True

if __name__ == '__main__':
    rospy.init_node("persons_status_plotter")
    subindexes = rospy.Subscriber("/next_trajectory", Empty,
                            new_trajectory, queue_size =1)
    sub = rospy.Subscriber("/safety_score", Float32,
                            callback, queue_size =1)
    sub2 = rospy.Subscriber("/safe_nav_vel", Twist,vel_callback, queue_size =1)
    while not rospy.is_shutdown():
        if not plot():
            time.sleep(0.1)
    rospy.spin()
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

    fig.suptitle("Velocity and Braking Profiles", fontsize=30)

        #axs[i].xlabel("Coordinate X [m]", fontsize=60)
        #axs[i].ylabel("Coordinate Y [m]", fontsize=60)    #    box = axs[i].get_position()
    #for ax in axs.flat:
    #SMALL_SIZE = 40
    #MEDIUM_SIZE = 48
    #BIGGER_SIZE = 56
    #plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    #plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    #plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    #plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    #plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    #plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    #axs[0].rcParams['axes.labelsize'] = 120

    #axs.flat[0].set(xlabel="", ylabel="Safe Speed [m/s]")
    #axs.flat[1].set(xlabel="Time", ylabel="Brake Command")
    #print type(axs.flat[0]), type(axs), type(fig)
    #plt.xlabel("Time [s]", fontsize=30)
    #axs[1].set(xlabel="Time", ylabel="Brake Command")
    #axs[0].set_xlabel('X axis', fontsize = 12)
    axs[0].set_ylabel("Speed [m/s]", fontsize = 20, rotation=90)
    axs[1].set_xlabel("Time [s]", fontsize = 20)
    axs[1].set_ylabel("Risk Factor", fontsize = 20, rotation=90)

    #axs[0].set_xlim([0, 1])
    axs[0].set_ylim([0, 1])
    axs[1].set_ylim([0, 1])

    #axs[0].set_xlim([0, 15.0])
    #axs[1].set_xlim([0, 15.0])
    #axs[0].set_yticks(np.linspace(0,1,0.1))
    axs[0].grid(True)
    axs[1].grid(True)

    #for i in range(2):
    #    axs[i].grid()#linestyle='-', linewidth=2)

    #    axs[i].set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    #    axs[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.legend()
    plt.savefig("speed_profile_sample.png")

    #rospy.sleep(10)
    #ax.show()
