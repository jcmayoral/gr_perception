import os
import sys
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy import ndimage
import sys

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print "use bagfile topic outputfolder"
        print sys.argv
        sys.exit()

    bagfilename = sys.argv[1]
    rostopic = sys.argv[2]
    outputfolder = sys.argv[3]

    print ('bagfile ', bagfilename)
    bag = rosbag.Bag(bagfilename)#'/example/2016-10-25-11-41-21_example.bag')
    #'/persons/2019-05-21-17-32-04.bag')#/2019-05-21-15-54-42.bag') #/grassrobotics/2019-05-21-17-00-12.bag.orig.active') #2019-05-21-16-48-08.bag.active')#'/long_video/2019-05-21-17-28-51.bag')
    bridge = CvBridge()

    if not os.path.exists(outputfolder):
        os.makedirs(outputfolder)
    os.chdir(outputfolder)

    for topic, msg, t in bag.read_messages(topics=rostopic):
        try:
            #print msg.encoding
            cv_image = bridge.imgmsg_to_cv2(msg)
            #cv_image = cv_image/255
            filename = os.path.join("image_"+str(msg.header.seq)+".jpg")
            cv2.imwrite(filename, cv_image)
        except CvBridgeError as e:
            print(e)
            exit(1)
    bag.close()
