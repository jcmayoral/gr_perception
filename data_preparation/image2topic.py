import cv2
import os
import sys
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

if __name__ == "__main__":
    rospy.init_node("image2topiC")
    bridge = CvBridge()
    pub = rospy.Publisher("/this_image", Image, queue_size=1)

    files_folder = sys.argv[1]
    print "FOLDER LOCATION ", files_folder
    for filename in os.listdir(files_folder):
        print filename
        print os.path.join(files_folder,filename)
        img = cv2.imread(os.path.join(files_folder,filename))
        image_message = bridge.cv2_to_imgmsg(img, encoding="bgr8")
        pub.publish(image_message)
        cv2.imshow("image", img)
        cv2.waitKey(100)
