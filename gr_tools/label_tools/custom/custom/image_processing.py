#/usr/bin/python
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import Image, CameraInfo
from gb_visual_detection_3d_msgs.msg import BoundingBoxes3d
import rospy
import actionlib
import copy
import cv2
from cv_bridge import CvBridge
import os
import tqdm
import numpy as np
import imutils
from skimage import img_as_float


def rotateImage(image, angle):
    return imutils.rotate(image, angle)
    row,col,channels = image.shape
    center=tuple(np.array([row,col])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col,row))
    return new_image

class ImageProcessing(object):
    def __init__(self, matches, depth_camera_info):
        self.depth_camera_info = depth_camera_info
        self.distance = 2.0
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        rospy.loginfo("Darknet Server found")
        self.odom_frame = "odom"
        self.image = Image()
        self.bbs = None
        self.img_pub = rospy.Publisher("/camera/depth/image_rect_raw2", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        self.bb_sub = rospy.Subscriber("/darknet_ros_3d/bounding_boxes", BoundingBoxes3d, self.bbs_cb)
        self.cv_bridge = CvBridge()

    def bbs_cb(self, msg):
        self.bbs = msg
        #print "BBs received"

    def run(self, matches):
        for rgbindex, depthindex in tqdm.tqdm(matches):
            #rgb_img = rotateImage(cv2.imread("image_{}.jpg".format(rgbindex)), 180)
            rgb_img = cv2.imread("image_{}.jpg".format(rgbindex))
            img_shape = rgb_img.shape
            rgb_img= self.cv_bridge.cv2_to_imgmsg(rgb_img,encoding="bgr8")
            #depth_img = rotateImage(cv2.imread("depthimage_{}.jpg".format(depthindex)), 180)
            #depth_img= self.cv_bridge.cv2_to_imgmsg(depth_img[:,:,2].reshape((480, 640,1)), encoding = "passthrough")
            depth_arr = np.load("depthimage_{}.npy".format(depthindex))
            depth_arr = np.asanyarray(depth_arr, dtype=np.uint16)*(255)
            depth_img= self.cv_bridge.cv2_to_imgmsg(depth_arr)
            #depth_img= self.cv_bridge.cv2_to_imgmsg(rotateImage(depth_arr,180))
            #print depth_img.encoding
            #print np.unique(depth_arr)

            self.depth_camera_info.header.seq = int(rgbindex)
            self.call(rgb_img, depth_img, int(rgbindex), img_shape)

    def call(self, image, depth_image, index, desired_shape):
        #print "Calling"
        goal = CheckForObjectsActionGoal()
        goal.goal.id = index
        goal.goal.image = image
        #print ("Called")
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(1.0))
        #print "result gotten"
        depth_image.header.seq = self.depth_camera_info.header.seq = index
        depth_image.header.stamp = self.depth_camera_info.header.stamp = rospy.Time.now()
        depth_image.header.frame_id = self.depth_camera_info.header.frame_id = "camera_depth_optical_frame"

        #print "publishing"
        for i in range(5):
            self.img_pub.publish(depth_image)
            self.depth_pub.publish(self.depth_camera_info)
            rospy.sleep(0.2)

        if self.bbs is None:
            return

        obbs = self.get_current_result().bounding_boxes.bounding_boxes
        if len(obbs) == 0:
            print "person not in image"
            return

        if len(obbs) != len(self.bbs.bounding_boxes):
            rospy.logwarn("error.. this will happen.. for now Skipping {} {}".format(len(obbs),len(self.bbs.bounding_boxes)))
            return

        for obb, bbs in zip(obbs, self.bbs.bounding_boxes):
            if obb.Class != "person":
                #print "Skipping " +bb.Class
                continue
            height, width, channels = desired_shape
            ring = min(3,int(bbs.zmin/self.distance))
            data = str(ring) + " "
            rx = obb.xmax - obb.xmin
            cx = float(rx/2+ obb.xmin)/width
            ry = obb.ymax - obb.ymin
            cy = float(ry/2 + obb.ymin)/height
            data += str(cx) + " "
            data += str(cy) + " "
            data += str(float(rx)/width) + " "
            data += str(float(ry)/height) + "\n"
            bbsx =[1,cx,cy,float(rx)/width,float(ry)/height]

            #print data, rx, ry
            #try:
            #    os.mkdir("labels")
            #except OSError as error:
            #pass

            label_filename = "image_{}.txt".format(index)

            with open(label_filename, "a+") as text_file:
                text_file.write(data)

            files_filename = "files.txt"
            with open(files_filename, "a+") as text_file:
                text_file.write(os.path.join(os.getcwd(),"image_{}.jpg".format(index))+"\n")



        #rospy.loginfo("Index {} obbs {} bbs {}".format(index,str(self.get_current_result()), str(self.bbs)))
        self.bbs = None


    def get_current_result(self):
        return self.client.get_result()

    def get_current_depth(self):
        return self.current_depth

    def callback_active(self):
        pass
        #rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        pass
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        #rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print (self.client.get_result())

    def callback_feedback(self,feedback):
        pass
        #rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    rospy.init_node('image_processing_label')
    imasin = ImageProcessing()
    rospy.spin()
