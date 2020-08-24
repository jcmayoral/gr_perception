#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray, RiskIndexes, RiskObject
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
indexes = RiskIndexes()


def callback(msg):
    global main_msg

    main_msg = msg

def callback_indexes(msg):
    global indexes
    indexes = msg

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


cmap = get_cmap(20,'hot')
cmapdyn = get_cmap(20,'hsv')

def plot():
    plt.cla()
    global main_msg, indexes
    global cmap, cmapdyn
    #rospy.logwarn(indexes.objects)

    items = dict()
    for k in indexes.objects:
        key,value = k.object_id, k.risk_index
        items[key] = value

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
    ncount1 = 5

    for p in main_msg.objects:
        nr = np.linalg.norm([p.pose.position.y, p.pose.position.x])
        print ("Radius ::: -> ", nr)
        #r.append(nr)
        narea = np.linalg.norm([p.speed.x, p.speed.y])*1000
        #area.append(narea)
        ntheta = np.arctan2(p.pose.position.y, p.pose.position.x)
        #theta.append(ntheta)
        if p.is_dynamic:
            ncolor = cmapdyn(ncount1)
        else:
            ncolor = cmap(ncount1)
        #else:
        #colors.append(ncolor)
        #    ncolor = 200
        #legens.append(p.object_id)
        risk = "UNKNOWN"
        if p.object_id in items.keys():
            risk = "WITH RISK:" +  str(items[p.object_id])
        ax.scatter(ntheta, nr, c=ncolor, cmap="coolwarm", s=narea, alpha=0.75, label=p.object_id+ risk)
        ncount1 = ncount1 + 1


    ax.set_ylim(0,20)
    #ax.set_yticks(np.arange(-20,20,5.0))

    #ax.scatter(theta, r, c=colors, s=area, cmap='hot', alpha=1.0)

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.canvas.draw()
    rospy.sleep(0.1)
    fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.close()




if __name__ == '__main__':
    rospy.init_node("persons_status_plotter")
    subindexes = rospy.Subscriber("/safety_indexes", RiskIndexes,
                            callback_indexes, queue_size =1)
    sub = rospy.Subscriber("/pointcloud_lidar_processing/found_object", FoundObjectsArray,
                            callback, queue_size =1)
    while not rospy.is_shutdown():
        plot()


    rospy.spin()
