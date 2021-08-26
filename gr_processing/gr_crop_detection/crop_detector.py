import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class MyParam:
    def __init__(self, default=0):
        self.value = default

    def on_change(self, value):
        self.value = value

    def get_value(self):
        return self.value


data = {
        #"gauss_filter":{
        #    "minval": 1,
        #    "maxval": 9,
        #    "default": 3
        #},
        "min_canny":{
            "minval": 0,
            "maxval": 255,
            "default": 50
        },
        "max_canny":{
            "minval": 0,
            "maxval": 255,
            "default": 150
        },
        "rho":{
            "minval": 1,
            "maxval": 10,
            "default": 5
        },
        "hough_threshold":{
            "minval": 1,
            "maxval": 100,
            "default": 50
        },
        "hough_min_len":{
            "minval": 1,
            "maxval": 200,
            "default": 20
        },
        "hough_gap":{
            "minval": 1,
            "maxval": 100,
            "default": 10
        }
}

class CropDetector:
    def __init__(self):
        rospy.init_node("gr_crop_detector")
        self.create_window()
        self.cv_bridge = CvBridge()
        self.initalized = True
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
        gf = 3##self.params["gauss_filter"].get_value()
        img_blur = cv2.GaussianBlur(img_gray, (gf, gf), 0)

        # perform edge detection
        min_canny = self.params["min_canny"].get_value()
        max_canny = self.params["max_canny"].get_value()

        img_edge = cv2.Canny(img_blur, threshold1=min_canny, threshold2=max_canny)

        # perform hough transform

        rho = self.params["rho"].get_value()
        hough_threshold = self.params["hough_threshold"].get_value()
        hough_min_line_len = self.params["hough_min_len"].get_value()
        hough_gap = self.params["hough_gap"].get_value()

        detected_lines = self.hough_lines_detection(img=img_edge,
                                               rho=rho,
                                               theta=np.pi / 180,
                                               threshold=hough_threshold,
                                               min_line_len=hough_min_line_len,
                                               max_line_gap=hough_gap)

        mask=color_image.copy()

        if detected_lines is not None:
            for line in detected_lines:
                for x1,y1,x2,y2 in line:
                    cv2.line(mask,(x1,y1),(x2,y2),255,1)

        #print(img_edge.shape, mask.shape)
        return img_edge#mask

    def create_window(self):
        cv2.namedWindow("process")
        self.params =  dict()
        for i,j in data.items():
            print ("track", i)
            self.params[i] = MyParam(j["default"])
            cv2.createTrackbar(i, "process", j["minval"], j["maxval"],self.params[i].on_change)
            cv2.setTrackbarPos(i, "process", self.params[i].get_value())



    def process_img(self, msg):
        #TODO CAMERA CALIBRATION
        original_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        h,w,c = original_image.shape
        out_image = original_image[h/2:h,w/2-200:w/2+200,:]
        out_image = self.get_lane_lines(out_image)
        cv2.imshow("process", out_image)
        cv2.waitKey(100)
