#!/usr/bin/env python

import math
from math import sin, cos, pi

import rospy
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, Vector3


class FakeOdom:
    def __init__(self):
        rospy.init_node('odometry_publisher')
        self.odom_broadcaster = tf.TransformBroadcaster()
        rospy.Subscriber("/odometry/base_raw", Odometry, self.odom_cb)
        print "setup done"
        rospy.spin()

    def odom_cb(self, msg):
        #odom_quat = tf.transformations.quaternion_from_euler(0, 0, th)
        # first, we'll publish the transform over tf
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        odom_quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z , msg.pose.pose.orientation.w] 
        self.odom_broadcaster.sendTransform((x, y, 0.),odom_quat,
            rospy.Time.now(),
            "base_link",
            "odom"
        )
        """
        # next, we'll publish the odometry message over ROS
        odom = Odometry()
        odom.header.stamp = current_time
        odom.header.frame_id = "odom"
        # set the position
        odom.pose.pose = Pose(Point(x, y, 0.), Quaternion(*odom_quat))
        # set the velocity
        odom.child_frame_id = "base_link"
        odom.twist.twist = Twist(Vector3(vx, vy, 0), Vector3(0, 0, vth))
        # publish the message
        odom_pub.publish(odom)
        last_time = current_time
        r.sleep()
        """


FakeOdom()
