#! /usr/bin/python
import rospy
from image_sim_animation import ImageSinAnimationLabeler
import tf2_ros
import tf2_geometry_msgs
import os
import sys
from cv_bridge import CvBridge
import cv2
import rosbag
import numpy as np
from test_bb import plot_bbs
from custom.utils import *
import numpy as np

def create_stamps_files(dbpath):
    bag =  rosbag.Bag(os.path.join(dbpath, "long_video" , "safecopy.bag"), 'r')
    rgb_info = extract_timestamps_frombag(bag, "/camera/color/image_raw")
    print "rgb", len(rgb_info[0]),  len(rgb_info[1])
    depth_info = extract_timestamps_frombag(bag, "/camera/depth/image_rect_raw")
    print "depth", len(depth_info[0]),  len(depth_info[1])
    bag.close()

    with open("rgb_info.txt", "w") as f:
        for seq, time,rtime in zip(rgb_info[0], rgb_info[1], rgb_info[2]):
            f.write("{} {} {}\n".format(seq,time,rtime))

    with open("depth_info.txt", "w") as f:
        for seq, time,rtime in zip(depth_info[0], depth_info[1], depth_info[2]):
            f.write("{} {} {}\n".format(seq,time, rtime))

def match_stamps(file1="rgb_info.txt", file2="depth_info.txt"):
    rgb_data = stamps_to_dict(open(file1,'r'))
    #rgb_data = sorted(rgb_data)
    depth_data = stamps_to_dict(open(file2,'r'))
    #depth_data = sorted(depth_data)
    matches = []
    for key, value in rgb_data.iteritems():
        min_index = -1
        val_minimum = 1000000000000000000
        for key2, value2 in depth_data.iteritems():
            time_diff = np.fabs(float(value)-float(value2))
            if time_diff < val_minimum:
                val_minimum = time_diff
                min_index = key2
        #print val_minimum
        if min_index >0:
            matches.append([key, min_index])

    matches = sorted(matches)
    save_matches(matches)

def store_imgs(storepath, rgb_topic = "/camera/color/image_raw", depth_topic= "/camera/depth/image_rect_raw"):
    if not os.path.exists(storepath):
        print "path does not exists"
    os.chdir(storepath)
    bag =  rosbag.Bag(os.path.join(dbpath, "long_video" , "safecopy.bag"), 'r')
    save_images(bag, depth_topic, True)
    save_images(bag, rgb_topic, True)


def execute(dbpath):
    depth_camera_info = extract_camera_info(os.path.join(dbpath, "camera" , "camera_info.bag"), "/camera/depth/camera_info")


if __name__ == '__main__':
    rospy.init_node('image_custom_manager')
    dbpath = "/media/datasets/nibio_summer_2019/"
    storepath = "./images"

    if len(sys.argv) == 1:
        print "use properly"
        sys.exit()
    if sys.argv[1] == "stamps":
        create_stamps_files(dbpath)
    if sys.argv[1] == "match":
        match_stamps()
    if sys.argv[1] == "store":
        store_imgs(storepath)
    print "FINISH"
