import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class CropDetector:
    def __init__(self):
        rospy.init_node("gr_crop_detector")
        self.cv_bridge = CvBridge()
        rospy.Subscriber("/camera/color/image_raw", Image, self.process_img)
        rospy.spin()

    def hough_lines_detection(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
        return lines

    def get_lane_lines(self,color_image):
        interm_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2HSV)
        interm_image[:,:,2] = 0

        # convert to grayscale
        img_gray = cv2.cvtColor(interm_image, cv2.COLOR_BGR2GRAY)

        # perform gaussian blur
        img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

        # perform edge detection
        img_edge = cv2.Canny(img_blur, threshold1=70, threshold2=255)

        # perform hough transform
        detected_lines = self.hough_lines_detection(img=img_edge,
                                               rho=4,
                                               theta=np.pi / 180,
                                               threshold=40,
                                               min_line_len=5,
                                               max_line_gap=1)

        mask=color_image.copy()

        if detected_lines is not None:
			for line in detected_lines:
				for x1,y1,x2,y2 in line:
					cv2.line(mask,(x1,y1),(x2,y2),255,1)

        print(img_edge.shape, mask.shape)
        return mask#cv2.vconcat([img_edge, mask])

    def process_img(self, msg):
        #TODO CAMERA CALIBRATION
        original_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        h,w,c = original_image.shape
        out_image = original_image[h/2:h,w/2-200:w/2+200,:]
        out_image = self.get_lane_lines(out_image)
        cv2.imshow("process", out_image)
        cv2.waitKey(100)
