#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time
import copy

plt.ion()
#fig, ax = plt.subplots(1,1,subplot_kw=dict(polar=True))
fig = plt.figure()
#ax = fig.add_subplot(111, projection='polar')
ax = fig.add_axes([0.1,0.1,0.8,0.8],polar=True)
#global main_msg

main_msg = FoundObjectsArray()

def callback(msg):
    global main_msg

    main_msg = msg

def plot():
    plt.cla()
    global main_msg
    #plt.cla()

    #N = 150
    """
    r = list()
    theta = list()
    area = list()
    colors = list()
    legens = list()
    """
    #ax.relim()
    #ax.autoscale_view()

    for p in main_msg.objects:
        nr = np.linalg.norm([p.pose.position.y, p.pose.position.x])
        #r.append(nr)
        narea = np.linalg.norm([p.speed.x, p.speed.y])*1000
        #area.append(narea)
        ntheta = np.arctan2(p.pose.position.y, p.pose.position.x)
        #theta.append(ntheta)
        ncolor = 25*nr
        if p.is_dynamic:
            #colors.append(128)
            ncolor *=10
        #else:
        #colors.append(ncolor)
        #    ncolor = 200
        #legens.append(p.object_id)
        ax.scatter(ntheta, nr, c=ncolor, s=narea, cmap='hot', alpha=0.75, label=p.object_id)

    ax.set_ylim(-20,20)
    ax.set_yticks(np.arange(-20,20,5.0))

    #ax.scatter(theta, r, c=colors, s=area, cmap='hot', alpha=1.0)

    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()    #fig.canvas.draw()
    rospy.sleep(0.05)
    #plt.close()




if __name__ == '__main__':
    rospy.init_node("persons_status_plotter")
    sub = rospy.Subscriber("/pointcloud_lidar_processing/found_object", FoundObjectsArray,
                            callback, queue_size =1)
    while not rospy.is_shutdown():
        plot()


    rospy.spin()
