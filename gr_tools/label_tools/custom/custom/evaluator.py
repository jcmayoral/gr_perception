#/usr/bin/python
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import Image
import rospy
import actionlib
import copy
import cv2
from cv_bridge import CvBridge
import os
import tqdm
import numpy as np
import imutils
import sys


def rotateImage(image, angle):
    return imutils.rotate(image, angle)
    row,col,channels = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

class ImageEvaluator(object):
    def __init__(self):
        #THIS IS DARKNET CLIENT
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        self.bridge = CvBridge()
        rospy.loginfo("Darknet Server found")

    def darket_call(self, image):
        print type(image)
        #print "Calling"
        ros_image = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        goal = CheckForObjectsActionGoal()
        goal.goal.id = 1
        goal.goal.image = ros_image
        #print ("Called")
        self.client.send_goal(goal.goal)
        flag = self.client.wait_for_result(rospy.Duration.from_sec(1.0))
        print self.client.get_result(), flag
        return self.client.get_result()
