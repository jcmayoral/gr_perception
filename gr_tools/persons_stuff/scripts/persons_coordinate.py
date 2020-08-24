#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray, RiskIndexes, RiskObject
import rospy
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
from tf.transformations import euler_from_quaternion

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
    fig.canvas.flush_events()    #fig.canvas.draw()
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
            risk = " RISK:" +  str(items[p.object_id])

        riskmsg = "::UNKNOWN"
        if p.object_id in items.keys():
            riskmsg = "::DANGER" if items[p.object_id] > 0.8 else "::SAFE"

        ax.scatter(ntheta, nr, c=ncolor, cmap="coolwarm", s=narea, alpha=0.75, label=p.object_id+ riskmsg)
        #angle = euler_from_quaternion([p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w])[2]

        """
        wrap = lambda x: x if x > 0 else 0
        print ntheta, type(ntheta), wrap(ntheta), type(ntheta)

        plt.annotate(p.object_id+riskmsg,
                    xy=(wrap(ntheta), nr),      # theta, radius
                    xytext=(0.05, 0.05),
                    xycoords='polar',
                    textcoords='polar',
                    arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='center',
                    verticalalignment='center',
                    clip_on=True)
        """
        ncount1 = ncount1 + 1

    ax.set_ylim(0,25)
    #ax.set_yticks(np.arange(-20,20,5.0))

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper center')
    fig.canvas.draw()
    #rospy.sleep(0.05)
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
