#!/usr/bin/env python

import math
from math import sin, cos, pi

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import PointCloud2, CameraInfo
from tf2_msgs.msg import TFMessage

class ReStamper:
    def __init__(self, in_topic, out_topic, msg_type, node_id):
        print in_topic
        rospy.init_node('republisher_'+"id_"+ node_id)
        if "tf" not in node_id:
            rospy.Subscriber(in_topic, msg_type, self.cb)
        else:
            rospy.Subscriber(in_topic, msg_type, self.tf_cb)
        self.pub = rospy.Publisher(out_topic, msg_type, queue_size=1)
        print "setup done"
        rospy.spin()

    def cb(self, msg):
        nmsg = msg
        nmsg.header.stamp = rospy.Time.now()
        self.pub.publish(nmsg)

    def tf_cb(self, msg):
        for m in msg.transforms:
            m.header.stamp = rospy.Time.now()
        nmsg = msg
        self.pub.publish(nmsg)

import sys
if sys.argv[1] == "color":
    ReStamper("/camera/color/camera_info", "/camera/color/camera_info2", CameraInfo, "color")
if sys.argv[1] == "depth":
    ReStamper("/camera/depth/camera_info", "/camera/depth/camera_info2", CameraInfo, "depth")
if sys.argv[1] == "tf":
    ReStamper("/tf_old", "/tf", TFMessage, "tf")
if sys.argv[1] == "tf_static":
    ReStamper("/tf_static_old", "/tf_static", TFMessage, "tf_static")
