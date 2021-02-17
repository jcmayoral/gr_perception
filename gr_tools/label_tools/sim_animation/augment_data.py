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
        #self.substractor = cv2.createBackgroundSubtractorMOG2(0,3.0, False)

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
        gray_image = cv2.cvtColor(oimage, cv2.COLOR_BGR2GRAY)
        newImage =cv2.imread("/home/jose/Pictures/field_test.jpg")
        onewImage =cv2.imread("/home/jose/Pictures/field_test.jpg")
        background =cv2.imread("/home/jose/Pictures/reference_sim.jpg")
        gray_backgroud = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
        newImage = cv2.resize(newImage,(oimage.shape[1], oimage.shape[0]))
        onewImage = cv2.resize(onewImage,(oimage.shape[1], oimage.shape[0]))
        #background = cv2.resize(background,(oimage.shape[1], oimage.shape[0]))

        nmask = cv2.subtract(gray_backgroud, gray_image) >5
        newImage[np.where(nmask)] = oimage[np.where(nmask)]

        #newImage = self.substractor.apply(oimage)
        print newImage.shape


        #playera
        #mask = cv2.inRange(oimage, (6,0., 9), (15, 50, 15))
        #print np.unique(mask)
        #newImage[np.where(mask)] = oimage[np.where(mask)]

        #piel
        #mask = cv2.inRange(newImage, (10, 8, 80), (60, 355, 200))
        #newImage[np.where(mask)] = onewImage[np.where(mask)]#[0,0,255]




        #cielo
        #mask = cv2.inRange(newImage, (0, 100, 80), (255, 255, 200))
        #newImage[np.where(mask)] = onewImage[np.where(mask)]


        #pasto
        #mask = cv2.inRange(oimage, (0,10, 0), (20,90, 100))
        #newImage[np.where(mask)] = onewImage[np.where(mask)]

        #pantalon
        mask = cv2.inRange(oimage, (26, 0, 0), (50, 30, 80))
        newImage[np.where(mask)] = [0,0,255]

        #playera
        mask = cv2.inRange(newImage, (4, 20, 0), (15, 50, 25))
        newImage[np.where(mask)] = [0,255,255]


        #whiteMask = cv2.bitwise_and(img2, mask) #created white mask
        #np.copyto(newImage, whiteMask)
        return newImage # cv2.cvtColor(newImage, cv2.COLOR_GRAY2BGR)

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
