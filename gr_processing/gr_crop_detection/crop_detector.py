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
        "v_crop_min":{
            "minval": 0,
            "maxval": 100,
            "default": 18
        },
        "v_crop_max":{
            "minval": 0,
            "maxval": 100,
            "default": 50
        },
        "h_crop_min":{
            "minval": 0,
            "maxval": 100,
            "default": 20
        },
        "h_crop_max":{
            "minval": 0,
            "maxval": 100,
            "default": 80
        },
        "min_canny":{
            "minval": 0,
            "maxval": 255,
            "default": 80
        },
        "max_canny":{
            "minval": 0,
            "maxval": 255,
            "default": 150
        },
        "rho":{
            "minval": 2,
            "maxval": 10,
            "default": 4
        },
        "hough_threshold":{
            "minval": 1,
            "maxval": 100,
            "default": 50
        },
        "hough_min_len":{
            "minval": 1,
            "maxval": 200,
            "default": 90
        },
        "hough_gap":{
            "minval": 1,
            "maxval": 100,
            "default": 10
        },
        "erosion_shape":{
            "minval": 0,
            "maxval": 2 ,
            "default": 1
        },
        "erosion_size":{
            "minval": 1,
            "maxval": 3,
            "default": 2
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
        img_erode = self.erosion(img_blur   )

        # perform edge detection
        min_canny = self.params["min_canny"].get_value()
        max_canny = self.params["max_canny"].get_value()

        img_edge = cv2.Canny(img_erode, threshold1=min_canny, threshold2=max_canny)

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

        mask=img_gray.copy()
        if detected_lines.shape[0] > 0:
            coordinates = np.zeros((detected_lines.shape[0], 4))
            print "COORD ", coordinates.shape

            if detected_lines is not None:
                for index,line in enumerate(detected_lines):
                    print index, np.array(line).shape, coordinates[index].shape
                    for x1,y1,x2,y2 in line:
                        cv2.line(mask,(x1,y1),(x2,y2),255,1)
                    coordinates[index] = np.asarray(line[0])
            avg_line = np.mean(coordinates,axis=0, dtype=np.uint)
            cv2.line(mask,(avg_line[0],avg_line[1]),(avg_line[2],avg_line[3]),255,10)

        #print(img_edge.shape, mask.shape)
        return cv2.hconcat([img_gray, img_edge, mask])#mask

    def create_window(self):
        cv2.namedWindow("process")
        self.params =  dict()
        for i,j in data.items():
            print ("track", i)
            self.params[i] = MyParam(j["default"])
            cv2.createTrackbar(i, "process", j["minval"], j["maxval"],self.params[i].on_change)
            cv2.setTrackbarPos(i, "process", self.params[i].get_value())

    def morph_shape(self, val):
        if val == 0:
            return cv2.MORPH_RECT
        elif val == 1:
            return cv2.MORPH_CROSS
        elif val == 2:
            return cv2.MORPH_ELLIPSE

    def erosion(self, image):

        erosion_size = self.morph_shape(self.params["erosion_size"].get_value())
        erosion_shape = self.morph_shape(self.params["erosion_shape"].get_value())
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
        return cv2.erode(image, element)


    def process_img(self, msg):
        #TODO CAMERA CALIBRATION
        original_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        h,w,c = original_image.shape
        v_crop_min = self.params["v_crop_min"].get_value()*w/100
        v_crop_max = self.params["v_crop_max"].get_value()*w/100
        h_crop_min = self.params["h_crop_min"].get_value()*w/100
        h_crop_max = self.params["h_crop_max"].get_value()*w/100


        out_image = original_image[v_crop_min:v_crop_max,h_crop_min:h_crop_max,:]
        print out_image.shape
        out_image = self.get_lane_lines(out_image)
        cv2.imshow("process", out_image)
        cv2.waitKey(100)
