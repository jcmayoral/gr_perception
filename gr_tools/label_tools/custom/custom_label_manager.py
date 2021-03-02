#! /usr/bin/python
import rospy
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
from custom.image_processing import ImageProcessing
import numpy as np

def create_stamps_files(matchfile_path, rgb_topic = "/camera/color/image_raw", depth_topic = "/camera/depth/image_rect_raw", bagfile_path="long_video/safecopy.bag"):
    bag =  rosbag.Bag(bagfile_path, 'r')
    rgb_info = extract_timestamps_frombag(bag, rgb_topic)
    print "rgb", len(rgb_info[0]),  len(rgb_info[1])
    depth_info = extract_timestamps_frombag(bag, depth_topic)
    print "depth", len(depth_info[0]),  len(depth_info[1])
    bag.close()
    os.chdir(matchfile_path)

    with open("rgb_info.txt", "w") as f:
        for seq, time,rtime in zip(rgb_info[0], rgb_info[1], rgb_info[2]):
            f.write("{} {} {}\n".format(seq,time,rtime))

    with open("depth_info.txt", "w") as f:
        for seq, time,rtime in zip(depth_info[0], depth_info[1], depth_info[2]):
            f.write("{} {} {}\n".format(seq,time, rtime))

def match_stamps(storepath, file1="rgb_info.txt", file2="depth_info.txt"):
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
        if min_index >0 and val_minimum < 2.0:
            matches.append([key, min_index])
        else:
            print time_diff

    matches = sorted(matches)
    save_matches(storepath,matches)

def store_imgs(storepath, bagfile, rgb_topic = "/camera/color/image_raw", depth_topic= "/camera/depth/image_rect_raw"):
    if not os.path.exists(storepath):
        print "path does not exists"
        return
    os.chdir(os.path.join(storepath))
    bag =  rosbag.Bag(bagfile,'r')#os.path.join(storepath, "safecopy.bag"), 'r')
    print storepath
    save_images(bag, depth_topic, True)
    save_images(bag, rgb_topic, False)


def execute(bagfile, storepath,matchfile="matches.txt", depth_info_topic="/camera/depth/camera_info"):
    matches = open(matchfile, "r").readlines()
    matches = [i.rstrip().split(" ") for i in matches]
    depth_camera_info = extract_camera_info(bagfile, depth_info_topic)
    #print depth_camera_info
    #matches = [[int(i), int(j)] for i,j in matches]
    os.chdir(storepath)
    proc = ImageProcessing(matches, depth_camera_info)
    proc.run(matches)

import argparse

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser("Required arguments")
    #my_parser.add_argument('action', help='What do you want to do?')
    sub_parsers = my_parser.add_subparsers(dest='command')

    stamps_parser = sub_parsers.add_parser('stamps')
    stamps_parser.add_argument("-matchfile_path", action="store", help="path to dataset folder", required = True)
    stamps_parser.add_argument("-bagfile", action="store", help="path to bagfile match", required = True)
    stamps_parser.add_argument("-rgb_topic", action="store", help="rgb image topic", required = True)
    stamps_parser.add_argument("-depth_topic", action="store", help="depth_image topic", required = True)

    match_parser = sub_parsers.add_parser('match')
    match_parser.add_argument("-storepath", action="store", help="path to store dataset folder", required = True)
    match_parser.add_argument("-rgb_matchfile", action="store", help="path to rgb match file", required = True)
    match_parser.add_argument("-depth_matchfile", action="store", help="path to depth match", required = True)

    store_parser = sub_parsers.add_parser('extract')
    store_parser.add_argument("-storepath", action="store", help="path to store dataset folder", required = True)
    store_parser.add_argument("-bagfile", action="store", help="path to bagfile match", required = True)
    store_parser.add_argument("-rgb_topic", action="store", help="rgb image topic", required = True)
    store_parser.add_argument("-depth_topic", action="store", help="depth_image topic", required = True)

    execute_parser = sub_parsers.add_parser('execute')
    execute_parser.add_argument("-storepath", action="store", help="path to store dataset folder", required = True)
    execute_parser.add_argument("-matchfile", action="store", help="path to match file match", required = True)
    execute_parser.add_argument("-bagfile", action="store", help="path to bagfile match", required = True)
    execute_parser.add_argument("-depth_info_topic", action="store", help="depth camera info topic", required = True)

    args = my_parser.parse_args()
    action = args.command

    #if len(sys.argv) == 1:
    #    print "use properly"
    #    sys.exit()
    if action == "stamps":
        #extract timestamps from topics in a bag and store them on a file
        matchfile_path = args.matchfile_path#"/home/jose/datasets/real_iros2021/"
        bagfile = args.bagfile
        rgb_topic = args.rgb_topic
        depth_topic = args.depth_topic
        create_stamps_files(matchfile_path, rgb_topic, depth_topic,  bagfile)
    if action == "match":
        #matches the two files created on stamp action
        storepath = args.storepath#"/home/jose/datasets/real_iros2021"
        rgb_matchfile = args.rgb_matchfile
        depth_matchfile = args.depth_matchfile
        match_stamps(storepath, rgb_matchfile, depth_matchfile)
    if action == "extract":
        #extact images from bagfile
        storepath = args.storepath#"/home/jose/datasets/real_iros2021"
        bagfile = args.bagfile
        rgb_topic = args.rgb_topic
        depth_topic = args.depth_topic
        store_imgs(storepath, bagfile, rgb_topic, depth_topic)
    if action == "execute":
        rospy.init_node('image_custom_manager')
        #macthes the timestamps of the match_file between depth and rgb
        storepath = args.storepath#"/home/jose/datasets/real_iros2021"
        bagfile = args.bagfile
        match_file = args.matchfile
        depth_info_topic = args.depth_info_topic
        execute(bagfile, storepath, match_file, depth_info_topic)
    print "FINISH"
