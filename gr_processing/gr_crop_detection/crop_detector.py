import rospy
import cv2
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import tf
import image_geometry

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
            "default": 0

#            "default": 18
        },
        "v_crop_max":{
            "minval": 0,
            "maxval": 100,
            #"default": 50
            "default": 100
        },
        "h_crop_min":{
            "minval": 0,
            "maxval": 100,
            #"default":30
            "default": 0
        },
        "h_crop_max":{
            "minval": 0,
            "maxval": 100,
            #"default": 80
            "default": 100
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
            "minval": 1,
            "maxval": 10,
            "default": 1
        },
        "hough_threshold":{
            "minval": 1,
            "maxval": 100,
            "default": 3
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
        self.center_coords = [None,None]
        self.tf_listener_ = tf.TransformListener()
        self.caminfo = CameraInfo()
        self.msgready = False
        self.cammodel = image_geometry.PinholeCameraModel()
        rospy.Subscriber("/zed/zed_node/rgb/camera_info", CameraInfo, self.info_cb)
        rospy.Subscriber("/zed/zed_node/rgb/image_rect_color", Image, self.process_img)
        rospy.spin()

    def hough_lines_detection(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
        return lines

    def info_cb(self,msg):
        self.caminfo = msg
        self.cammodel.fromCameraInfo(msg)
        self.msgready = True

    def transform_and_mark_poses(self,img):
        coords = list()

        if self.tf_listener_.frameExists("base_link") and self.tf_listener_.frameExists("zed_left_camera_optical_frame"):
            rospy.logerr("WORKING")
            for i in range(5):
                t = self.tf_listener_.getLatestCommonTime("base_link", "zed_left_camera_optical_frame") #TODO REPLACE PROPER LINKS
                p1 = PoseStamped()
                p1.header.frame_id = "base_link"
                p1.pose.position.x = 1.0*i
                p1.pose.position.z = 1.0
                #p1.pose.position.y = -1.0*i
                p1.pose.orientation.w = 1.0    # Neutral orientation
                #transform = tf.StampedTransform()
                position, rotation = self.tf_listener_.lookupTransform("zed_left_camera_optical_frame", "base_link", t);
                p_in_base = self.tf_listener_.transformPose("zed_left_camera_optical_frame", p1)
                if self.msgready:
                    #rospy.logerr(self.cammodel.intrinsicMatrix())
                    #a = self.cammodel.project3dToPixel([p_in_base.pose.position.x, p_in_base.pose.position.y, p_in_base.pose.position.z])
                    a = self.cammodel.project3dToPixel([1.0,0.0+1*i,-10-(i*2.0)])

                    print a
                    coords.append((int(a[0]), int(a[1])))
                    #cv2.circle(img,(int(p_in_base.pose.position.x), int(p_in_base.pose.position.y)), 10,255,4)

        for i in range(1,len(coords)-1):
            print "LINE ", coords[i], " to " , coords[i+1]
            cv2.line(img, coords[i], coords[i+1], (255,0,0), 10)

        print img.shape


        return img

    def update_and_draw_center(self, img, mx,my):
        if self.center_coords[0] is None:
            self.center_coords[0] = mx
            self.center_coords[1] = my
        print "UPDATED COORDS B ", self.center_coords, mx,my,  self.center_coords[0] - float(np.fabs(self.center_coords[0] - float(mx))/img.shape[0])* mx
        #self.center_coords[0]= (0.2*(self.center_coords[0] + float(mx)/self.center_coords[0]))
        self.center_coords[0] = self.center_coords[0] - 0.1*float(np.fabs(self.center_coords[0] - float(mx))/img.shape[0])* mx
        self.center_coords[1]= self.center_coords[1] - 0.1*float(np.fabs(self.center_coords[1] - float(my))/img.shape[1])* my
        print "UPDATED COORDS ", self.center_coords, mx,my
        cv2.circle(img,(int(self.center_coords[0]), int(self.center_coords[1])), 20,0,10)


    def get_lane_lines(self,color_image):
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
        img_erode = self.transform_and_mark_poses(color_image)
        #FOR # DEBUG:
        print "return"
        return img_erode

        if detected_lines is None:
            print "ERROR "
            return cv2.hconcat([img_erode, img_edge, mask])#mask

        if detected_lines.shape[0] > 0:
            coordinates = np.zeros((detected_lines.shape[0], 4))
            print "COORD ", coordinates.shape

            if detected_lines is not None:
                for index,line in enumerate(detected_lines):
                    print index, np.array(line)#.shape, coordinates[index].shape
                    for x1,y1,x2,y2 in line:
                        cv2.line(mask,(x1,y1),(x2,y2),255,1)
                        cv2.circle(mask,(x1+(x2-x1)/2,y1+(y2-y1)/2), 10,125,10)

                        #cv2.line(mask,(0,y1),(img_gray.shape[0],y2),255,1)
                        #cv2.line(mask,(x1,0),(x2, img_gray.shape[0]),255,1)
                    coordinates[index] = np.asarray(line[0])

            avg_line = np.mean(coordinates,axis=0, dtype=np.uint)
            print "AVGS", avg_line
            minx = min(avg_line[0], avg_line[2])
            maxx = max(avg_line[0], avg_line[2])
            miny = min(avg_line[1], avg_line[3])
            maxy = max(avg_line[1], avg_line[3])
            mx = int(minx+(maxx - minx)/2)
            my = int(miny+(maxy - miny)/2)
            print "MS", mx,my
            if mx < 0 or my < 0:
                print "less than 0"
                return
            self.update_and_draw_center(mask,mx,my)
            #cv2.line(mask,(avg_line[0],0),(avg_line[2],mask.shape[1]),255,10)


        #TEST
        # Find contours
        cnts = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        # Iterate thorugh contours and draw rectangles around contours
        mean_cnts = list()
        for c in cnts:
            x,y,w,h = cv2.boundingRect(c)
            mean_cnts.append([x,y,w,h])
        avg_cnts = np.mean(mean_cnts, axis=0, dtype=np.uint)
        max_cnts = np.max(mean_cnts, axis=0)
        min_cnts = np.min(mean_cnts, axis=0)
        range_cnts =  max_cnts - min_cnts

        #print "MIN : ", min_cnts
        #print "MAX : ", max_cnts
        #print "AVG : ", avg_cnts
        #print "RANGE : ", range_cnts
        cv2.circle(mask,(min_cnts[0]+min_cnts[2],min_cnts[1]+min_cnts[3]), 20,255,10)
        cv2.circle(mask,(max_cnts[0]+max_cnts[2],max_cnts[1]+max_cnts[3]), 20,255,10)

        cv2.rectangle(mask,( min_cnts[0], 0), (max_cnts[0]+ max_cnts[2], max_cnts[3]), 255, 2)

        #print(img_edge.shape, mask.shape)
        return cv2.hconcat([img_erode, img_edge, mask])#mask

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
