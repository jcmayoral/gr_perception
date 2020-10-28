#!/usr/bin/python3
from safety_msgs.msg import FoundObjectsArray, RiskIndexes, RiskObject
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


cmap = get_cmap(20,'binary')
cmapdyn = get_cmap(20,'hsv')

def plot():
    fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.cla()
    global main_msg, indexes
    global cmap, cmapdyn
    #rospy.logwarn(indexes.objects)
    colors = dict()
    colors["DANGER"] = 'k'
    colors["WARNING"] = 'r'
    colors["SAFE"] = 'g'
    colors["UNKNOWN"] = 'y'


    items = dict()
    for k in indexes.objects:
        key,value = k.object_id, k.risk_index
        items[key] = value

    narea=80
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
        x = p.pose.position.x
        y = p.pose.position.y

        #if p.is_dynamic:
        #    ncolor = cmapdyn(ncount1)
        #else:
        #    ncolor = cmap(ncount1)

        risk = "UNKNOWN"
        #if p.object_id in items.keys():
        #   risk = " RISK:" +  str(items[p.object_id])

        riskmsg = "UNKNOWN"
        if not p.object_id in items:
            break
        if p.object_id in items.keys():
            if 0.2 < items[p.object_id] < 0.8:
                riskmsg = "WARNING"
            if items[p.object_id] > 0.8:
                riskmsg = "DANGER"
            if items[p.object_id] < 0.2:
                riskmsg = "SAFE"
        print x,y, riskmsg
        ax.scatter(x, y,c=cmap(int(20*items[p.object_id])), s=narea+10.0, alpha=0.75)#, label=riskmsg)
        #ax.scatter(x, y,cmap='viridis',  c=cmap(int(20*items[p.object_id])), s=narea+10.0, alpha=0.75)#, label=riskmsg)
        #angle = euler_from_quaternion([p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w])[2]
        ncount1 = ncount1 + 1

    print "a"
    fig.canvas.draw()

    #ax.set_ylim(0,15)
    #rospy.sleep(0.1)
    #plt.close()




if __name__ == '__main__':
    rospy.init_node("persons_status_plotter")
    subindexes = rospy.Subscriber("/safety_indexes", RiskIndexes,
                            callback_indexes, queue_size =1)
    sub = rospy.Subscriber("/pointcloud_lidar_processing/found_object", FoundObjectsArray,
                            callback, queue_size =1)
    while not rospy.is_shutdown():
        plot()
    #rospy.spin()
    print "out"
    #plt.show()
    plt.rc('font', size=20) 
    plt.rc('axes', titlesize=40) 
    plt.rc('axes', labelsize=20) 
    rect = patches.Rectangle((-1,-0.75),1.8,1.5)
    ax.add_patch(rect)
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

    plt.grid(linestyle='-', linewidth=2)
    plt.xlabel("Coordinate X [m]", fontsize=30)
    plt.ylabel("Coordinate Y [m]", fontsize=30)
    plt.title("Risk Evaluation Map")
    #  plt.legend()
    plt.savefig("range_of_data_risk_zones_safetyindex.png")
    print "END"
    #rospy.sleep(10)
    #ax.show()
