import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class CropDetector:
    def __init__(self):
        rospy.init_node("gr_crop_detector")
        self.cv_bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.process_img)
        rospy.spin()

    def process_img(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        cv2.imshow("process", cv_image)
        cv2.waitKey(100)
