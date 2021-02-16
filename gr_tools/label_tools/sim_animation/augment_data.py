#! /usr/bin/python
import rospy
import tf2_ros
import tf2_geometry_msgs
import os
import sys
from cv_bridge import CvBridge
import cv2
import numpy as np
from test_bb import plot_bbs
import tqdm

class DatasetAugmenter:
    def __init__(self, dbpath, depth = False, version = 1000, start_count = 0):
        #super(ImageSinAnimationLabeler, self).__init__()
        #super(PersonSimAnimation, self).__init__()
        self.count = start_count
        self.backward_motion = False
        self.initialize = False
        self.target_frame = "camera_link"
        self.odom_frame = "odom"
        #self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
        #self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.bridge = CvBridge()
        self.distance = 2.0
        self.seq = 0

        try:
            os.chdir(dbpath)
        except:
            print("error in folder" + dbpath)
            sys.exit()

        try:
            os.chdir("depth_testdataset_v"+str(version))
        except:
            print("error in folder")
            sys.exit()
        cv2.namedWindow('test')
        cv2.setMouseCallback('test',self.mouseRGB)


    def mouseRGB(self,event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
            colorsB = self.img[y,x,0]
            colorsG = self.img[y,x,1]
            colorsR = self.img[y,x,2]
            colors = self.img[y,x]
            print("Red: ",colorsR)
            print("Green: ",colorsG)
            print("Blue: ",colorsB)
            print("BRG Format: ",colors)
            print("Coordinates of pixel: X: ",x,"Y: ",y)

    def clean_image(self,oimage):
        image = np.zeros(oimage.shape)
        #cielo
        #mask = cv2.inRange(oimage, (0, 100, 80), (255, 255, 200))
        #whiteImage = np.zeros_like(mask)
        #image[np.where(mask)] = [255,255,255]

        #playera
        mask = cv2.inRange(oimage, (6,0., 9), (15, 50, 15))
        whiteImage = np.zeros_like(mask)
        image[np.where(mask)] = [0,0,255]

        #piel
        mask = cv2.inRange(oimage, (10, 8, 80), (60, 355, 200))
        image[np.where(mask)] = [0,0,255]


        #pantalon
        mask = cv2.inRange(oimage, (30, 0, 0), (50, 50, 80))
        #whiteImage = np.zeros_like(mask)
        image[np.where(mask)] = [0,0,255]


        #pasto
        #mask = cv2.inRange(oimage, (0,50, 0), (20,90, 100))
        #whiteImage = np.zeros_like(mask)
        #image[np.where(mask)] = [255,255,255]



        #playera
        #mask = cv2.inRange(image, (6, 26, 9), (11, 32, 15))
        #whiteImage = np.zeros_like(mask)
        #image[np.where(mask)] = [0,0,255]



        #whiteMask = cv2.bitwise_and(whiteImage, mask) #created white mask
        #np.copyto(newImage, whiteMask)
        return image # cv2.cvtColor(newImage, cv2.COLOR_GRAY2BGR)

    def run(self):
        with open("files.txt", "r") as text_file:
            for file in tqdm.tqdm(text_file):
                #print file
                self.img =cv2.imread(file.rstrip())
                self.img = self.clean_image(self.img)
                cv2.imshow("test", self.img)
                cv2.waitKey(500)

if __name__ == '__main__':
    dbpath = "/media/datasets/simanimation/"
    startcount=920
    manager = DatasetAugmenter(dbpath, depth=True, version = 3, start_count = startcount)
    #rospy.logerr("image request " + str(i) )
    manager.run()
    #rospy.spin()
