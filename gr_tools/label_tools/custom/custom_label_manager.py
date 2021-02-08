#! /usr/bin/python
import rospy
from image_sim_animation import ImageSinAnimationLabeler
from person_sim_animation import PersonSimAnimation
import tf2_ros
import tf2_geometry_msgs
import os
import sys
from cv_bridge import CvBridge
import cv2
import rosbag
import numpy as np
from test_bb import plot_bbs
from custom_dataset.utils import extract_camera_info, extract_timestamps, extract_timestamps_frombag


def create_stamps_files(dbpath):
    bag =  rosbag.Bag(os.path.join(dbpath, "long_video" , "safecopy.bag"), 'r')
    rgb_info = extract_timestamps_frombag(bag, "/camera/color/image_raw")
    print "rgb", len(rgb_info[0]),  len(rgb_info[1])
    depth_info = extract_timestamps_frombag(bag, "/camera/depth/image_rect_raw")
    print "depth", len(depth_info[0]),  len(depth_info[1])
    bag.close()

    with open("rgb_info.txt", "w") as f:
        for seq, time in zip(rgb_info[0], rgb_info[1]):
            f.write("{} {}\n".format(seq,time))

    with open("depth_info.txt", "w") as f:
        for seq, time in zip(depth_info[0], depth_info[1]):
            f.write("{} {}\n".format(seq,time))


def execute(dbpath):
    depth_camera_info = extract_camera_info(os.path.join(dbpath, "camera" , "camera_info.bag"), "/camera/depth/camera_info")


if __name__ == '__main__':
    rospy.init_node('image_custom_manager')
    dbpath = "/media/datasets/nibio_summer_2019/"
    if len(sys.argv) == 1:
        print "use properly"
        sys.exit()
    if sys.argv[1] == "stamps":
        create_stamps_files(dbpath)

    print "FINISH"
