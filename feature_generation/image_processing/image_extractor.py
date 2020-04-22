import os
import sys
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from scipy import ndimage

PATH_IN="/media/datasets/thermal_fieldsafe/"
PATH_OUT="/home/jose/ros_ws/src/gr_perception/feature_generation/rgb_2_thermal/results"
IMAGE_TOPICS=["/FlirA65/image_raw", "/Multisense/depth", "/Multisense/left/image_rect_color", "/Multisense/right/image_rect"]

bagfiles=["filter_bag0", "filter_bag1", "filter_bag2", "filter_bag3", "filter_bag4", "filter_bag5"]

if __name__ == '__main__':

    for name in bagfiles:
        print ('bagfile ', name)
        bag = rosbag.Bag(PATH_IN+name+'.bag')#'/example/2016-10-25-11-41-21_example.bag')
        #'/persons/2019-05-21-17-32-04.bag')#/2019-05-21-15-54-42.bag') #/grassrobotics/2019-05-21-17-00-12.bag.orig.active') #2019-05-21-16-48-08.bag.active')#'/long_video/2019-05-21-17-28-51.bag')
        bridge = CvBridge()

        if not os.path.exists(PATH_OUT):
            os.makedirs(PATH_OUT)

        for topic, msg, t in bag.read_messages(topics=IMAGE_TOPIC):
            try:
                print msg.header.seq
                cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
                #cut the tractor from raw
                #cv_image = cv_image[:,300:, :]
                key = cv2.waitKey(50)
                label = topic.replace("/", "_")
                store_path = os.path.join(PATH_OUT, topic)
                #print (store_path)
                if not os.path.exists(store_path):
                    os.makedirs(store_path)
                filename = os.path.join(store_path ,"image_"+name+"_"+ str(msg.header.seq)+".jpg")
                #print (filename)
                cv2.imwrite(filename, cv_image)
            except CvBridgeError as e:
                print(e)
                exit(1)
        print ("closing" + name)
        bag.close()
