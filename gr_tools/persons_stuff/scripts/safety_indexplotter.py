#!/usr/bin/python3
from safety_msgs.msg import RiskIndexes
import rospy
import matplotlib.pyplot as plt
import numpy as np
from collections import deque


plt.ion()
#plt.set_ylabel('Y [mm]')
#plt.set_title('NAILS surface')
#plt.set_xlabel('X [mm]')
#plt.show(block=False)
fig, ax = plt.subplots()
#ax1 = fig.add_subplot(121)
#ax1.hold(True)
#plt.show(block=False)
#plt.draw()
hl, = ax.plot([], [],'o')

message_read = False

xdata = deque(list(),10)
ydata = deque(list(),10)

def callback(msg):
    #plt.clf()
    global message_read
    #X = np.arange(-508, 510, 203.2)
    #Y = np.arange(-508, 510, 203.2)
    #X, Y = np.meshgrid(X, Y)
    X = np.arange(10)*np.random.rand()
    Y = np.arange(10)* np.random.rand()
    xdata.append(X)
    ydata.append(Y)
    message_read = True
    #plot(X,Y)

def plot():
    #plt.draw()
    #ax1.scatter(X, Y)
    print ("plot")
    print (len(xdata))
    hl.set_xdata(xdata)
    hl.set_ydata(ydata)
    ax.relim()
    ax.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.draw()
    rospy.sleep(0.5)




if __name__ == '__main__':
    rospy.init_node("safety_plotter")
    sub = rospy.Subscriber("/safety_indexes", RiskIndexes,
                            callback, queue_size =1)


    while not rospy.is_shutdown():
        while not message_read:
            pass
        plot()
        message_read = False
    #plt.show()
    #rospy.spin()
