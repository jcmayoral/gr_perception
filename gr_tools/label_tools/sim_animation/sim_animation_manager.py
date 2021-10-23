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
import numpy as np
from test_bb import plot_bbs
import tqdm
import sys

class SimAnimationManager(ImageSinAnimationLabeler, PersonSimAnimation):
    def __init__(self, dbpath, depth = False, version = 1000, start_count = 0, class_id=0):
        #super(ImageSinAnimationLabeler, self).__init__()
        #super(PersonSimAnimation, self).__init__()
        self.count = start_count
        PersonSimAnimation.__init__(self, class_id = int(class_id))
        ImageSinAnimationLabeler.__init__(self, depth)
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
            os.chdir(dbpath)
        except:
            print("error in folder" + dbpath)
            sys.exit()

        try:
            os.mkdir(os.path.join("v"+ str(version),class_id))
        except:
            pass

        try:
            os.chdir(os.path.join("v"+ str(version),class_id))
        except:
            print("error in folder")
            sys.exit()

        print "SIM ANIMATION MANAGER CREATEd"

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
        #rospy.loginfo("New Feedback:%s" % str(feedback))
        if not self.initialize:
            self.initialize = True
            return
        self.backward_motion =feedback.backward
        #print (self.backward_motion)


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

        #cv2.waitKey()
        transform_pose = self.transform()
        #rows cols
        height, width, channels = cv_image.shape
        #print (height, width)
        flag = False

        bbs = result.bounding_boxes
        bbsx = []

        for bb in bbs.bounding_boxes:
            if bb.Class != "person":
                #print "Skipping " +bb.Class
                continue
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
            bbsx =[1,cx,cy,float(rx)/width,float(ry)/height]

            #print data, rx, ry
            label_filename = os.path.join(os.getcwd(),str(self.count),self.folder_name[int(self.backward_motion)], "image_"+ str(self.seq)+".txt")

            with open(label_filename, "a+") as text_file:
                text_file.write(data)

        if flag:
            #plot_bbs(cv_image, bbsx, visualize=True)
            #return
            #TO SAVE
            cv2.imwrite(filename, cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB) )
            #append image to master file
            with open("files.txt", "a+") as text_file:
                text_file.write(filename+"\n")
            if self.depth:
                depth_filename = os.path.join(os.getcwd(),str(self.count),self.folder_name[int(self.backward_motion)], "depthimage_"+ str(self.seq)+".jpg")
                np_filename = os.path.join(os.getcwd(),str(self.count),self.folder_name[int(self.backward_motion)], "depthimage_"+ str(self.seq)+".npy")
                cv_depth_image = self.bridge.imgmsg_to_cv2(self.current_depth, desired_encoding='passthrough')
                #cv_depth_image = np.asarray(cv_depth_image, dtype=np.uint8)
                #cv_image_norm = cv2.normalize(cv_depth_image, None, 0, 2, cv2.NORM_L1)
                #cv_image_array = np.array(cv_depth_image, dtype = np.uint8)
                #cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
                #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_depth_image, alpha=0.03), cv2.COLORMAP_JET)
                #print np.unique(cv_image_norm, return_counts=True)
                #print cv_image_norm.shape
                #print cv_image_norm.shape
                #
                #cv_image_norm = cv_image_norm.reshape(480,640,3)
                depth_image = np.asanyarray(cv_depth_image)
                cv_image_norm = cv2.normalize(depth_image, depth_image, 0, 255, cv2.NORM_MINMAX)
                #cv2.imshow("Depth", cv_image_norm)
                np.save(np_filename, depth_image)

                #cv2.waitKey()
                cv2.imwrite(depth_filename, cv_image_norm)
        self.seq = self.seq + 1
        self.is_processing = False
        rospy.sleep(0.50)

if __name__ == '__main__':
    rospy.init_node('image_sim_manager')
    dbpath = "/home/jose/datasets/simulation_white_october2021/"
    startcount=1
    manager = SimAnimationManager(dbpath, depth=False, version = 5, start_count = startcount, class_id = sys.argv[1])
    endcount = 1000
    for i in tqdm.tqdm(range(startcount,endcount)):
        #rospy.logerr("image request " + str(i) )
        manager.run()
    #rospy.spin()
