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


data = {
        #"gauss_filter":{
        #    "minval": 1,
        #    "maxval": 9,
        #    "default": 3
        #},
        "v_crop_min":{
            "minval": 0,
            "maxval": 100,
#            "default": 0
            "default": 18
        },
        "v_crop_max":{
            "minval": 0,
            "maxval": 100,
            "default": 50
#            "default": 100
        },
        "h_crop_min":{
            "minval": 0,
            "maxval": 100,
            "default":40
#            "default": 0
        },
        "h_crop_max":{
            "minval": 0,
            "maxval": 100,
            "default": 90
#            "default": 100
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

        self.tfBuffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tfBuffer)
        #self.tf_listener_ = tf.TransformListener()
        self.caminfo = CameraInfo()
        self.msgready = False
        self.cammodel = image_geometry.PinholeCameraModel()
        self.local_path = Path()
        rospy.Subscriber("/move_base_flex/diff/global_plan", Path, self.path_cb)
        rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.info_cb)
        rospy.Subscriber("/camera/color/image_raw", Image, self.process_img)
        rospy.spin()

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
        coords = list()
        h,w,c = img.shape

        v_crop_min = self.params["v_crop_min"].get_value()*w/100
        v_crop_max = self.params["v_crop_max"].get_value()*w/100
        h_crop_min = self.params["h_crop_min"].get_value()*h/100
        h_crop_max = self.params["h_crop_max"].get_value()*h/100

        for p in self.local_path.poses:
            p1 = p

            transform = self.tfBuffer.lookup_transform('camera_depth_link',p.header.frame_id, rospy.Time())
            p_in_base = tf2_geometry_msgs.do_transform_pose(p, transform)

            if self.msgready:
                a = self.cammodel.project3dToPixel([p_in_base.pose.position.x, p_in_base.pose.position.y, p_in_base.pose.position.z])
                if any(a) < 0:
                    print ("SKip point of path")
                    pass

                coords.append((int(a[0]), int(a[1])))
                cv2.circle(img,(int(a[0]), int(a[1])), 10,127,4)
                #cv2.circle(img,(int(v_crop_min + a[0]), int(h_crop_min + a[1])), 10,255,4)

        """
        colors = [0,0,0]
        s = 0
        for i in range(1,len(coords)-1):
            print( "LINE ", coords[i], " to " , coords[i+1])
            colors[s] = 255
            s = s +1
            if s >2:
                s =0
            cv2.line(img, coords[i], coords[i+1], tuple(colors), 10)

        print( img.shape)
        """

        return img

    def update_and_draw_center(self, img, mx,my, color=(255,255,0)):
        if True:#self.center_coords[0] is None:
            self.center_coords[0] = mx
            self.center_coords[1] = my
        #print( "UPDATED COORDS B ", self.center_coords, mx,my,  self.center_coords[0] - float(np.fabs(self.center_coords[0] - float(mx))/img.shape[0])* mx)
        #self.center_coords[0]= (0.2*(self.center_coords[0] + float(mx)/self.center_coords[0]))
        self.center_coords[0] += 0.1*float(np.fabs(self.center_coords[0] - float(mx))/mx)* mx
        self.center_coords[1] += 0.1*float(np.fabs(self.center_coords[1] - float(my))/mx)* my
        #print( "UPDATED COORDS ", self.center_coords, mx,my)
        cv2.circle(img,(int(self.center_coords[0]), int(self.center_coords[1])), 10,color,10)

    def draw_line(self, img, line, slope, h_min, v_min):
        print( "AVGS", line)
        minx = min(line[0], line[2])
        maxx = max(line[0], line[2])
        miny = min(line[1], line[3])
        maxy = max(line[1], line[3])
        mx = int(minx+(maxx - minx)/2)
        my = int(miny+(maxy - miny)/2)

        if mx < 0 or my < 0:
            print( "less than 0")
            return


        print("M " , mx + h_min , my + v_min)
        start_y = int(my + v_min)
        #slope = float(maxx-minx)/(maxy-miny)
        start_x = -int(slope*start_y) + int(maxx)# = int(h_min)
        print(start_x, start_y)


        h,w,c = img.shape

        py= int(miny) #int(minx)
        px=int((py)*slope) + int(minx)
        ##ending point
        #qx=w
        #qy=-(x2-w)*m+y2


        end_y = int(maxy)
        end_x = int(slope*end_y) + int(maxx)
        print (end_x, end_y, px,py)
        print("SLOPE ", slope)

        cv2.line(img, (end_x + h_min, end_y+v_min), (px + h_min, py + v_min), (127,0,255), 10)

        self.update_and_draw_center(img,h_min + mx, v_min + my, color=(0,0,255))


    def get_lane_lines(self,original_img):

        h,w,c = original_img.shape
        v_crop_min = self.params["v_crop_min"].get_value()*w/100
        v_crop_max = self.params["v_crop_max"].get_value()*w/100
        h_crop_min = self.params["h_crop_min"].get_value()*h/100
        h_crop_max = self.params["h_crop_max"].get_value()*h/100

        color_image = original_img.copy()[v_crop_min:v_crop_max,h_crop_min:h_crop_max,:]

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

        mask=img_edge.copy()
        output_image = self.transform_and_mark_poses(original_img.copy())
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
            #print( "COORD ", coordinates.shape)

            if detected_lines is not None:
                for index,line in enumerate(detected_lines):
                    #print( index, np.array(line))
                    for x1,y1,x2,y2 in line:
                        cv2.line(mask,(x1,y1),(x2,y2),255,1)
                        cv2.circle(mask,(x1+(x2-x1)/2,y1+(y2-y1)/2), 10,125,10)
                        cv2.circle(output_image,(  h_crop_min + x1+(x2-x1)/2,  v_crop_min + y1+(y2-y1)/2), 10,125,10)
                        slope = float(x2-x1)/(y2-y1)
                        if slope >0:
                            print ("right index {} slope{}".format(index,slope))
                            r_mean_slope += [slope]
                            right_slopes.append(line[0])
                        else:
                            print ("left index {} slope{}".format(index,slope))
                            l_mean_slope += [slope]
                            left_slopes.append(line[0])
                        #cv2.line(mask,(0,y1),(img_gray.shape[0],y2),255,1)
                        #cv2.line(mask,(x1,0),(x2, img_gray.shape[0]),255,1)
                    coordinates[index] = np.asarray(line[0])

            avg_line = np.mean(coordinates,axis=0, dtype=np.uint)

            l_avg_line = np.mean(left_slopes,axis=0, dtype=np.uint)
            r_avg_line = np.mean(right_slopes,axis=0, dtype=np.uint)

            self.draw_line(output_image, r_avg_line, np.mean(r_mean_slope), h_crop_min, v_crop_min)
            self.draw_line(output_image, l_avg_line,  np.mean(l_mean_slope), h_crop_min, v_crop_min)


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

        #print( "MIN : ", min_cnts)
        #print( "MAX : ", max_cnts)
        #print( "AVG : ", avg_cnts)
        #print( "RANGE : ", range_cnts)
        cv2.circle(mask,(min_cnts[0]+min_cnts[2],min_cnts[1]+min_cnts[3]), 20,255,10)
        cv2.circle(mask,(max_cnts[0]+max_cnts[2],max_cnts[1]+max_cnts[3]), 20,255,10)

        cv2.rectangle(mask,( min_cnts[0], 0), (max_cnts[0]+ max_cnts[2], max_cnts[3]), 255, 2)

        full_mask = np.zeros(output_image.shape, dtype=np.uint8)
        print(img_edge.shape, mask.shape, output_image.shape)
        """
        v_crop_min = self.params["v_crop_min"].get_value()*w/100
        v_crop_max = self.params["v_crop_max"].get_value()*w/100
        h_crop_min = self.params["h_crop_min"].get_value()*h/100
        h_crop_max = self.params["h_crop_max"].get_value()*h/100

        color_image = original_img.copy()[v_crop_min:v_crop_max,h_crop_min:h_crop_max,:]
        """
        full_mask[v_crop_min:v_crop_max,h_crop_min:h_crop_max,0] = mask
        full_mask[v_crop_min:v_crop_max,h_crop_min:h_crop_max,1] = mask
        full_mask[v_crop_min:v_crop_max,h_crop_min:h_crop_max,2] = mask

        #return cv2.hconcat([output_image, full_mask])#mask
        return cv2.bitwise_or(output_image, full_mask)#cv2.hconcat([output_image, img_edge, mask])#mask
        return cv2.bitwise_and(mask, img_edge)#cv2.hconcat([output_image, img_edge, mask])#mask

    def create_window(self):
        cv2.namedWindow("process")
        self.params =  dict()
        for i,j in data.items():
            #print ("track", i)
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
        out_image = self.get_lane_lines(original_image)
        cv2.imshow("process", out_image)
        cv2.waitKey(100)
