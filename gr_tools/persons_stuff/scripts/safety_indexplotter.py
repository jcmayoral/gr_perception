#!/usr/bin/python3
from safety_msgs.msg import RiskIndexes
import rospy
import matplotlib.pyplot as plt
import numpy as np
from collections import deque, defaultdict, namedtuple
from itertools import repeat
import time
import copy

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
#l, = ax.plot([], [],'o')

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
            graphs[m.object_id] = PlotData(0.0,deque(list(),100))

        #print(type(time.perf_counter()))
        #print(type(graphs[m.object_id].x))
        graphs[m.object_id].y.append(m.risk_index)
        graphs[m.object_id] = PlotData(time.perf_counter(), graphs[m.object_id].y)

        #graphs[m.object_id].x = time.perf_counter()

    message_read = True
    #plot(X,Y)

def plot():
    #plt.draw()
    #ax1.scatter(X, Y)
    plt.cla()
    for g in graphs.keys():
        #hl.set_xdata(np.arange(len(graphs[g].y)))
        #hl.set_ydata(graphs[g].y)
        xdata= np.arange(100)#len(graphs[g].y))
        ydata = list(repeat(0,100 -len(graphs[g].y)))
        ydata.extend(list(graphs[g].y))

        #if len(ydata)>100:
        #    print(ydata)
        #ydata = graphs[g].y
        ax.plot(xdata,ydata, label=g)
    ax.relim()
    ax.autoscale_view()
    ax.legend()
    fig.canvas.draw()
    fig.canvas.flush_events()    #fig.canvas.draw()
    #plt.draw()
    #rospy.sleep(0.05)


def clear():
    tic = time.perf_counter()

    tmpg = copy.deepcopy(graphs)
    for g in tmpg.keys():
        if tic - tmpg[g].x >3:
            del graphs[g]



if __name__ == '__main__':
    rospy.init_node("safety_plotter")
    sub = rospy.Subscriber("/safety_indexes", RiskIndexes,
                            callback, queue_size =1)


    while not rospy.is_shutdown():
        while not message_read:
            pass
        plot()
        clear()
        message_read = False
    #plt.show()
    #rospy.spin()
