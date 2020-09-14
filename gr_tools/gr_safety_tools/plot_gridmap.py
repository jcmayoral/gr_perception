#!/usr/bin/python

import rosbag
import numpy as np
from grid_map_msgs.msg import GridMap

import matplotlib.pyplot as plt

rb = rosbag.Bag("resources/2020-09-14-12-07-08.bag")
c = 0
for topic, msg, t in rb.read_messages(topics="/grid_map"):
    cells_x =  int(msg.info.length_x/msg.info.resolution)
    cells_y =  int(msg.info.length_y/msg.info.resolution)
    #assuming map frame in origin
    layers = np.asarray(msg.data)
    for layer,nlayer in zip(layers, msg.layers):
        for i in layer.layout.dim:
            print i.label, i.size, i.stride
        data = np.asarray(layer.data).reshape(cells_x,cells_y)
        #print data.shape
        plt.figure()
        plt.title(nlayer)
        plt.imshow(data, cmap='Wistia')
        plt.colorbar()
        #plt.show()
        plt.savefig("images/"+nlayer+str(c)+".jpg")
    c = c +1
