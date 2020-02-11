#!/usr/bin/python2
import rospy
import rosbag
#import tf2_ros
#import tf2_geometry_msgs

import sys
import numpy as np
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Header
import argparse
import time
import os
import numpy as np

help_text="This script stores detection results"

def create_folder(class_name,dataset="new"):
    try:
        os.makedir(os.path.join(dataset,class_name))
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = help_text)
    parser.add_argument("--bag", "-b", help="set input bagname")
    parser.add_argument("--group", "-g", default="safety_labelsnibio_2019")
    parser.add_argument("--topic", "-t", default="/camera/color/image_raw")
    parser.add_argument("--debug", "-d", default=1)

    args = parser.parse_args()
    debug_mode = bool(int(args.debug))
    bag = rosbag.Bag(args.bag, mode="r")
    #recoder = DetectionRecorder(folder=args.group)
    f = open(os.path.join(args.group,"events_recorded"),"r")
    stamp = f.readline()
    #print stamp.strip('\n')
    stamp = float(stamp.strip('\n'))
    print (stamp)
    bridge = CvBridge()
    current_state = -1
    classes_dict = {0:"Danger", 1: "UnSafe", 2:"Warning", -1: "Safer"}
    str_state = classes_dict[current_state]
    print("current state " , current_state)
    for topic, msg, t in bag.read_messages(topics=args.topic):
        image_stamp = msg.header.stamp
        #print(stamp)
        #print(image_stamp.to_sec(), stamp)
        cv_image = bridge.imgmsg_to_cv2(msg)
        cv2.imshow(str_state, cv_image)
        cv2.waitKey(25)

        if image_stamp.to_sec() > stamp:
            print(stamp, image_stamp.to_sec())
            r = open(os.path.join(args.group,str(stamp)),"r")
            print(r)
            if len(nstr) != 0:
                r2 = r.readline()
                r2 = r2.split(',')
                print(r2[1])
                current_state = int(r2[1])
                print("current state ", current_state)
                if current_state > 2:
                    current_state = -1

                #print(msg.header.seq)
                #TODO READ FILE
                #TODO visualize
                stamp = float(f.readline().rstrip('\n'))
                str_state = classes_dict[current_state]
            r.close()






        #while not recoder.message_processed:
            #rospy.sleep(0.2)
        #    pass
    f.close()
    bag.close()
