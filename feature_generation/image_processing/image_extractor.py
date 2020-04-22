import os
import sys
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy import ndimage
import numpy as np

PATH_IN="/media/datasets/thermal_fieldsafe/"
PATH_OUT="/media/datasets/thermal_fieldsafe/dataset"
IMAGE_TOPICS=["/FlirA65/image_raw", "/Multisense/depth", "/Multisense/left/image_rect_color", "/Multisense/right/image_rect"]

bagfiles=["filter_bag0", "filter_bag1", "filter_bag2", "filter_bag3", "filter_bag4", "filter_bag5"]

import matplotlib.pyplot as plt
from PIL import Image

if __name__ == '__main__':

    if not os.path.exists(PATH_OUT):
        os.makedirs(PATH_OUT)

    for name in bagfiles:
        print ('bagfile ', name)
        bag = rosbag.Bag(PATH_IN+name+'.bag')#'/example/2016-10-25-11-41-21_example.bag')
        #'/persons/2019-05-21-17-32-04.bag')#/2019-05-21-15-54-42.bag') #/grassrobotics/2019-05-21-17-00-12.bag.orig.active') #2019-05-21-16-48-08.bag.active')#'/long_video/2019-05-21-17-28-51.bag')
        bridge = CvBridge()

        for topic, msg, t in bag.read_messages(topics=IMAGE_TOPICS):
            try:
                #print msg.header.stamp.to_sec(), msg.header.stamp.to_time(), msg.header.stamp.to_nsec()
                label = topic.replace("/", "_")
                store_path = os.path.join(PATH_OUT, label)
                cv_image = bridge.imgmsg_to_cv2(msg)

                if not os.path.exists(store_path):
                    os.makedirs(store_path)

                if "16" in msg.encoding:
                    filename = os.path.join(store_path ,"image_"+name+"_"+ str(msg.header.stamp.to_nsec())+".tiff")
                    im = Image.fromarray(cv_image)
                    im.save(filename)
                else:
                    filename = os.path.join(store_path ,"image_"+name+"_"+ str(msg.header.stamp.to_nsec())+".jpg")
                    cv2.imwrite(filename, cv_image)
            except CvBridgeError as e:
                print(e, " in topic", topic)
                continue
        print ("closing" + name)
        bag.close()
