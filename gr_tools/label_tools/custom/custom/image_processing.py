#/usr/bin/python
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import rospy
import actionlib
import copy
import cv2
from cv_bridge import CvBridge
import os
import tqdm

class ImageProcessing(object):
    def __init__(self, matches, depth_camera_info):
        self.depth_camera_info = depth_camera_info
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        rospy.loginfo("Darknet Server found")
        self.odom_frame = "odom"
        self.image = Image()
        self.id = 0
        self.img_pub = rospy.Publisher("/camera/depth/image_rect_raw2", Image, queue_size=1)
        self.depth_pub = rospy.Publisher("/camera/depth/camera_info", CameraInfo, queue_size=1)
        #pos_sub = Subscriber("/animated_human/location", Vector3Stamped, queue_size=10)
        self.cv_bridge = CvBridge()

    def run(self, matches):
        for rgbindex, depthindex in tqdm.tqdm(matches):
            rgb_img= self.cv_bridge.cv2_to_imgmsg(cv2.imread("image_{}.jpg".format(rgbindex)),encoding="rgb8")
            depth_img= self.cv_bridge.cv2_to_imgmsg(cv2.imread("depthimage_{}.jpg".format(rgbindex)))
            self.depth_camera_info.header.seq = int(rgbindex)
            self.call(rgb_img, depth_img, int(rgbindex))

    def call(self, image, depth_image, index):
        print "Calling"
        goal = CheckForObjectsActionGoal()
        goal.goal.id = index
        goal.goal.image = image
        #print ("Called")
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(1.0))
        print "result gotten"
        depth_image.header.seq = self.depth_camera_info.header.seq = index
        depth_image.header.stamp = self.depth_camera_info.header.stamp = rospy.Time.now()
        print "publishing"
        self.img_pub.publish(depth_image)
        self.depth_pub.publish(self.depth_camera_info)
        rospy.sleep(2)
        pc2 = rospy.wait_for_message("/points", PointCloud2, timeout=None)



    def get_current_result(self):
        return self.client.get_result()

    def get_current_depth(self):
        return self.current_depth

    def callback_active(self):
        pass
        #rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print (self.client.get_result())

    def callback_feedback(self,feedback):
        rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    rospy.init_node('image_processing_label')
    imasin = ImageProcessing()
    rospy.spin()
