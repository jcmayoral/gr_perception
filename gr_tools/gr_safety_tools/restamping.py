#!/usr/bin/env python

import math
from math import sin, cos, pi

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3
from sensor_msgs.msg import PointCloud2


class ReStamper:
    def __init__(self):
        rospy.init_node('republisher')
        rospy.Subscriber("/velodyne_points", PointCloud2, self.cb)
        self.pub = rospy.Publisher("/velodyne_points/restamped", PointCloud2, queue_size=1)
        print "setup done"
        rospy.spin()

    def cb(self, msg):
        nmsg = msg
        nmsg.header.stamp = rospy.Time.now()
        self.pub.publish(nmsg)


ReStamper()
