#!/usr/bin/python3
from safety_msgs.msg import RiskIndexes
import rospy
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict, namedtuple


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

PlotData = namedtuple('Point', ['x', 'y'])
graphs = defaultdict(PlotData)
#graphs = list()

def callback(msg):
    #plt.clf()
    global message_read
    global graphs

    for m in msg.objects:
        if m.object_id not in graphs.keys():
            graphs[m.object_id] = PlotData(deque(list(),100),deque(list(),100))
        graphs[m.object_id].x.append(2)
        graphs[m.object_id].y.append(m.risk_index)
    message_read = True
    #plot(X,Y)

def plot():
    #plt.draw()
    #ax1.scatter(X, Y)
    for g in graphs.keys():
        print(type(hl))
        hl.set_xdata(np.arange(len(graphs[g].y)))
        hl.set_ydata(graphs[g].y)
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
