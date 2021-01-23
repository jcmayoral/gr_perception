#! /usr/bin/python
import rospy
from image_sim_animation import ImageSinAnimationLabeler
from person_sim_animation import PersonSimAnimation
import tf2_ros
import tf2_geometry_msgs
import os
import sys
from cv_bridge import CvBridge
import cv2

class SimAnimationManager(ImageSinAnimationLabeler, PersonSimAnimation):
    def __init__(self):
        #super(ImageSinAnimationLabeler, self).__init__()
        #super(PersonSimAnimation, self).__init__()
        self.count = 0
        PersonSimAnimation.__init__(self)
        ImageSinAnimationLabeler.__init__(self)
        self.backward_motion = False
        self.initialize = False
        self.target_frame = "camera_link"
        self.odom_frame = "odom"
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.folder_name = ["Forward", "Backward"]
        self.bridge = CvBridge()

        try:
            os.mkdir("testdataset")
        except:
            pass

        try:
            os.chdir("testdataset")
        except:
            print("error in folder")
            sys.exit()

        try:
            for f in self.folder_name:
                os.mkdir(f)
        except:
            pass

    def run(self):
        self.person_call()
        self.count = self.count + 1
        self.backward_motion = False
        self.initialize = False

    def transform(self):
        transform = self.tf_buffer.lookup_transform(self.target_frame,
                                       self.person_pose.header.frame_id, #source frame
                                       rospy.Time(0), #get the tf at first available time
                                       rospy.Duration(1.0)) #wait for 1 second
        return tf2_geometry_msgs.do_transform_vector3(self.person_pose, transform)

    #Overridehas_turned
    def pcallback_feedback(self,feedback):
        rospy.loginfo("New Feedback:%s" % str(feedback))
        print (self.initialize)
        if not self.initialize:
            print ("first")
            self.initialize = True
            return
        self.backward_motion =feedback.backward
        print (self.backward_motion)


    def callback_done(self,state, result):
        #rospy.logwarn("new image bounding boxes for %s " %str(self.backward_motion))
        #rospy.logwarn("person pose %s " %str(self.transform()))
        filename = os.path.join(os.getcwd(),self.folder_name[int(self.backward_motion)], "image_"+ str(result.id)+".jpg")
        cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='passthrough')
        cv2.imwrite(filename, cv_image)

if __name__ == '__main__':
    rospy.init_node('image_sim_manager')
    manager = SimAnimationManager()

    for i in range(2):
        rospy.logerr("image request " + str(i) )
        manager.run()
    #rospy.spin()
