#!/usr/bin/env python

import math
from math import sin, cos, pi

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import PointCloud2, CameraInfo, Image
from std_msgs.msg import Header
from tf2_msgs.msg import TFMessage
import cv_bridge

class Convertor:
    def __init__(self, in_topic, out_topic, msg_type, node_id):
        print in_topic
        self.cv_bridge = cv_bridge.CvBridge()
        self.header = Header()
        rospy.init_node('convertor_'+"id_"+ node_id)
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.h_cb)
        rospy.Subscriber(in_topic, msg_type, self.cb)
        self.pub = rospy.Publisher(out_topic, msg_type, queue_size=1)
        self.pub2 = rospy.Publisher("/camera/depth/camera_info2", CameraInfo, queue_size=1)
        print "setup done"
        rospy.spin()

    def cb(self, msg):
        try:
            depth_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        except CvBridgeError, e:
            print e

        ros_img = self.cv_bridge.cv2_to_imgmsg(depth_image)#,encoding="passtrough")
        ros_img.header.frame_id = msg.header.frame_id
        ros_img.header= self.header.header
        self.pub.publish(ros_img)
        self.pub2.publish(self.header)

    def h_cb(self, msg):
        print msg
        self.header = msg

import sys
Convertor("/camera/depth/image_rect_raw", "/camera/depth/image_rect_raw2", Image, "depth")
