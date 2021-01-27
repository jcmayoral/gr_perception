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
        self.distance = 2.0
        self.seq = 0

        try:
            os.mkdir("testdataset")
        except:
            pass

        try:
            os.chdir("testdataset")
        except:
            print("error in folder")
            sys.exit()

    def create_folders(self, path):
        try:
            for f in self.folder_name:
                os.mkdir(os.path.join(path,f))
        except:
            pass

    def run(self):
        self.person_call()
        self.count = self.count + 1
        self.seq = 0
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
        if not self.initialize:
            self.initialize = True
            return
        self.backward_motion =feedback.backward
        print (self.backward_motion)


    def callback_done(self,state, result):
        try:
            os.mkdir(str(self.count))
        except:
            pass
        #rospy.logwarn("new image bounding boxes for %s " %str(self.backward_motion))
        #rospy.logwarn("person pose %s " %str(self.transform()))
        self.create_folders(str(self.count))
        filename = os.path.join(os.getcwd(),str(self.count),self.folder_name[int(self.backward_motion)], "image_"+ str(self.seq)+".jpg")
        cv_image = self.bridge.imgmsg_to_cv2(self.image, desired_encoding='passthrough')

        transform_pose = self.transform()
        #rows cols
        height, width, channels = cv_image.shape
        #print (height, width)
        flag = False

        bbs = result.bounding_boxes
        for bb in bbs.bounding_boxes:
            flag = True
            ring = min(3,int(transform_pose.vector.x/self.distance))
            data = str(ring) + " "
            rx = bb.xmax -bb.xmin
            cx = float(rx/2+ bb.xmin)/width
            ry = bb.ymax -bb.ymin
            cy = float(ry/2 + bb.ymin)/height
            data += str(cx) + " "
            data += str(cy) + " "
            data += str(float(rx)/width) + " "
            data += str(float(ry)/height) + "\n"
            print data, rx, ry
            label_filename = os.path.join(os.getcwd(),str(self.count),self.folder_name[int(self.backward_motion)], "image_"+ str(self.seq)+".txt")

            with open(label_filename, "a+") as text_file:
                text_file.write(data)

        if flag:
            cv2.imwrite(filename, cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) )
            with open("files.txt", "a+") as text_file:
                text_file.write(filename+"\n")
        self.seq = self.seq + 1



if __name__ == '__main__':
    rospy.init_node('image_sim_manager')
    manager = SimAnimationManager()

    for i in range(15):
        rospy.logerr("image request " + str(i) )
        manager.run()
    #rospy.spin()
