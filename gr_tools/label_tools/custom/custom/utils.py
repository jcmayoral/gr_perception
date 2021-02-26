from sensor_msgs.msg import CameraInfo
from collections import OrderedDict
import cv_bridge
import tqdm
import numpy as np
import cv2
import rosbag
from image_processing import rotateImage
import os

def match_timestamps(rgb_stamps, depth_stamps):
    if len(rgb_stamps) < len(depth_stamps):
        return

def extract_camera_info(bagfile, info_topic):
    msg = None
    with rosbag.Bag(bagfile, 'r') as rbag:
        msgs = rbag.read_messages(info_topic)
        #print outbag.get_type_and_topic_info()
        for i, m in enumerate(msgs):
            #print m[0], type(m[1])
            msg = m[1]
    return msg
            #print i,type(CameraInfo(m[1]))

def extract_timestamps_frombag(rbag, info_topic):
    stamps = []
    seqs = []
    rtimes = []
    stamps = []
    start_time = None
    rstart_time = None
    #with rosbag.Bag(bagfile, 'r') as rbag:
    msgs = rbag.read_messages(info_topic)
    #print outbag.get_type_and_topic_info()
    for i, m in enumerate(msgs):
        #print m[0], type(m[1])
        if start_time is None:
            start_time = m[1].header.stamp.to_sec()
            rstart_time = m[2]
        seqs.append(m[1].header.seq)
        stamps.append(m[1].header.stamp.to_sec()- start_time)
        rtimes.append(m[2] - rstart_time)
    return seqs, stamps, rtimes

def extract_timestamps(bagfile, info_topic):
    stamps = []
    start_time = None
    rstart_time = None

    with rosbag.Bag(bagfile, 'r') as rbag:
        msgs = rbag.read_messages(info_topic)
        #print outbag.get_type_and_topic_info()
        for i, m in enumerate(msgs):
            #print m[0], type(m[1])
            if start_time is None:
                start_time = m[1].header.stamp.to_sec
                rstart_time = m[2]
            stamps.append(m[1].header.stamp.to_sec()-start_time)
    return stamps

def save_matches(storepath, matches):
    with open(os.path.join(storepath,"matches.txt"), "w") as f:
        for m in matches:
            f.write("{} {} \n".format(m[0],m[1]))

def stamps_to_dict(data):
    mydict = dict()
    for i in data:
        seq, stamp, rt = i.rstrip().split(" ")
        mydict[int(seq)] = stamp
    return OrderedDict(sorted(mydict.items(), key=lambda t: t[0]))

def save_images(rbag, info_topic, is_depth=False):
    msg = None
    bridge = cv_bridge.CvBridge()
    for topic, msg,t in tqdm.tqdm(rbag.read_messages(info_topic)):
        if is_depth:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #cv_image = rotateImage(cv_image, 180)
            filename = "depthimage_"+str(msg.header.seq)+".jpg"
            np_filename = "depthimage_"+ str(msg.header.seq)+".npy"
            depth_image = np.asanyarray(cv_image, dtype=float)/(255)

            cv_image_norm = cv2.normalize(depth_image, depth_image, 0, 255, cv2.NORM_MINMAX)
            #cv_image_norm = rotateImage(cv_image_norm, 180)
            #cv2.imshow("Depth", cv_image_norm)
            #print depth_image
            np.save(np_filename, cv_image_norm)
            cv2.imwrite(filename, cv_image_norm)
            #cv2.imwrite(filename, cv_image)
        else:
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            #cv_image = rotateImage(cv_image, 180)
            filename = "image_"+str(msg.header.seq)+".jpg"
            cv2.imwrite(filename, cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) )
