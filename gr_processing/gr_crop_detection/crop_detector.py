import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from cv_bridge import CvBridge
import numpy as np
import tf
import image_geometry
import tf2_ros
import tf2_geometry_msgs


class MyParam:
    def __init__(self, default=0):
        self.value = default

    def on_change(self, value):
        self.value = value

    def get_value(self):
        return self.value

import yaml

data = yaml.load(open("config.yaml"), Loader=yaml.Loader)

class CropDetector:
    def __init__(self):
        rospy.init_node("gr_crop_detector")
        self.create_window()
        self.cv_bridge = CvBridge()
        self.initalized = True
        self.center_coords = [None,None]

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        #self.tf_listener_ = tf.TransformListener()
        self.caminfo = CameraInfo()
        self.msgready = False
        self.cammodel = image_geometry.PinholeCameraModel()
        self.local_path = Path()
        self.load_roipoints()
        rospy.Subscriber("/move_base_flex/diff/global_plan", Path, self.path_cb)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_cb)
        rospy.Subscriber("/camera/color/image_raw", Image, self.process_img)
        rospy.spin()

    def load_roipoints(self):
        myroi = list()
        #TODO FIND A BETTER WAY TO STANDARDIZE
        for i, j in data['roi_points'].items():
            myroi.append([j[0],j[1]])
        self.myROI = np.array(myroi)


    def hough_lines_detection(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
        return lines

    def path_cb(self, path):
        self.local_path = path

    def info_cb(self,msg):
        self.caminfo = msg
        self.cammodel.fromCameraInfo(msg)
        self.msgready = True

    def transform_and_mark_poses(self,img):
        indeces =  np.linspace(0,len(self.local_path.poses)-1,10, dtype=np.uint16)

        if len(self.local_path.poses) < 3:
            return img
        #for p in self.local_path.poses:
        for ind,ind2 in zip(indeces[:-2],indeces[1:-3]):
            print ind, ind2, len(self.local_path.poses)
            frame = self.local_path.poses[ind].header.frame_id
            p1 = self.local_path.poses[ind]
            p2 = self.local_path.poses[ind2]

            self.transform_point(img,p1,p2)

        return img

    def transform_point(self,img, p1, p2, frame="map"):
        transform = self.tfBuffer.lookup_transform('camera_depth_link',frame, rospy.Time())
        p_base1 = tf2_geometry_msgs.do_transform_pose(p1, transform).pose.position
        p_base2 = tf2_geometry_msgs.do_transform_pose(p2, transform).pose.position

        if self.msgready:
            self.mark_path(img,[p_base1.x, p_base1.y, p_base1.z], [p_base2.x, p_base2.y, p_base2.z] )


    def mark_path(self,img, pix1, pix2):
        coords1 = self.cammodel.project3dToPixel(pix1)
        coords2 = self.cammodel.project3dToPixel(pix2)

        if any(coords1) < 0 or any(coords2) < 0:
            print ("SKip point of path")
            return

        print (coords1, coords2)
        cv2.line(img, (int(coords1[0]) ,int(coords1[1])), (int(coords2[0]), int(coords2[1])), 255, 10)

        cv2.circle(img,(int(coords1[0]), int(coords1[1])), 10,127,1)


    def draw_line(self, img, line, color =(0,0,0)):

        #If we dont receive at least to points polyfit will fail
        if line.shape[0] < 2:
            print("NO line detected")
            return

        #split values
        x = line[:,0]
        y = line[:,1]

        #Fitting line
        polyline = np.polyfit(y, x, 3)#, full=True)
        #First point
        #If set to zeros intersection point
        y0 = 200
        #evaluate
        x0 = int(np.polyval(polyline,y0))

        #last point down
        y1 = img.shape[0]
        #evaluate
        x1 = int(np.polyval(polyline,y1))

        cv2.line(img, (x0 ,y0), (x1, y1), color, 10)

    def roi(self,img):
        mask = np.zeros(img.shape, dtype=np.uint8)
        cv2.fillPoly(mask,  np.int32([self.myROI]), 255)

        # returning the image only where mask pixels are nonzero
        return cv2.bitwise_and(img, mask)

    def get_lane_lines(self,original_img):

        h,w,c = original_img.shape

        color_image = original_img.copy()
        #Set value to 0 in HSV
        interm_image = cv2.cvtColor(color_image,cv2.COLOR_RGB2HSV)
        interm_image[:,:,2] = 0

        # convert to grayscale
        img_gray = cv2.cvtColor(interm_image, cv2.COLOR_BGR2GRAY)

        # perform gaussian blur
        gf = 3##self.params["gauss_filter"].get_value()
        img_blur = cv2.GaussianBlur(img_gray, (gf, gf), 0)
        img_erode = self.erosion(img_blur)

        # perform edge detection
        min_canny = self.params["min_canny"].get_value()
        max_canny = self.params["max_canny"].get_value()

        img_edge = cv2.Canny(img_erode, threshold1=min_canny, threshold2=max_canny)
        img_edge = self.roi(img_edge)

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

        #mask=np.zeros(img_edge.shape)#fimg_edge.copy()

        output_image = self.transform_and_mark_poses(color_image.copy())
        #FOR # DEBUG:
        #return output_image

        if detected_lines is None:
            print( "ERROR ")
            return output_image#cv2.hconcat([img_erode, mask])#mask

        left_slopes = list()
        right_slopes = list()
        r_mean_slope = list()
        l_mean_slope = list()

        if detected_lines.shape[0] > 0:
            coordinates = np.zeros((detected_lines.shape[0], 4))

            if detected_lines is not None:
                for index,line in enumerate(detected_lines):
                    for x1,y1,x2,y2 in line:
                        cv2.line(output_image,(x1,y1),(x2,y2),(255,0,127),4)
                        cv2.circle(output_image,(x1,y1), 10,255,10)
                        cv2.circle(output_image,(x2,y2), 10,255,10)

                        slope = float(x2-x1)/(y2-y1)
                        if slope >0.1:
                            print ("right index {} slope{}".format(index,slope))
                            r_mean_slope += [slope]
                            right_slopes.append([x1,y1])
                            right_slopes.append([x2,y2])

                        elif slope <0.1:
                            print ("left index {} slope{}".format(index,slope))
                            l_mean_slope += [slope]
                            left_slopes.append([x1,y1])
                            left_slopes.append([x2,y2])
                        #cv2.line(mask,(0,y1),(img_gray.shape[0],y2),255,1)
                        #cv2.line(mask,(x1,0),(x2, img_gray.shape[0]),255,1)
                    coordinates[index] = np.asarray(line[0])

            avg_line = np.mean(coordinates,axis=0, dtype=np.uint)

            l_avg_line = np.mean(left_slopes,axis=0, dtype=np.uint)
            r_avg_line = np.mean(right_slopes,axis=0, dtype=np.uint)

            self.draw_line(output_image, np.asarray(right_slopes), (255,0,255))
            self.draw_line(output_image, np.asarray(left_slopes),  (0,255,0))


        #TEST
        # Find contours
        im2, cnts, hierarchy = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(output_image, cnts,-1, 127,5)


        return output_image

    def create_window(self):
        cv2.namedWindow("process")
        self.params =  dict()
        for i,j in data.items():
            #Avoid ROI TODO find a better way
            print i
            if "roi" in i:
                continue
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

        enable = self.params["erosion_enable"].get_value()
        if enable != 1:
            print ("Erosion is disable")
            return image

        erosion_size = self.morph_shape(self.params["erosion_size"].get_value())
        erosion_shape = self.morph_shape(self.params["erosion_shape"].get_value())
        element = cv2.getStructuringElement(erosion_shape, (2 * erosion_size + 1, 2 * erosion_size + 1),
                                       (erosion_size, erosion_size))
        return cv2.erode(image, element)


    def process_img(self, msg):
        #TODO CAMERA CALIBRATION
        original_image = self.cv_bridge.imgmsg_to_cv2(msg, "bgr8")
        out_image = self.get_lane_lines(original_image)
        cv2.imshow("process", out_image)
        cv2.waitKey(100)
