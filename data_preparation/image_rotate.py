z#/usr/bin/python
import rospy
import cv2
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

import os

def msg_callback(cv_image):
    (rows,cols,channels) = cv_image.shape
    M = cv2.getRotationMatrix2D((rows/2, cols/2), 180, 1)
    cv_image = cv2.warpAffine(cv_image, M, (rows, cols))

    #cv2.imshow("Image window", cv_image)
    #cv2.waitKey(3)

if __name__ == '__main__':
    for i in os.walk("/media/datasets/NMBUSUMMER021/ok-31111221"):
        print i
        0
