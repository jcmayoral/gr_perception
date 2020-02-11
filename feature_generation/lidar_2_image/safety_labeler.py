#!/usr/bin/python
import rospy
import rosbag
import tf2_ros
import tf2_geometry_msgs
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, PoseStamped
from std_msgs.msg import Header
import argparse
import time
import os
import numpy as np

help_text="This script stores detection results"

class DetectionRecorder:
    def __init__(self, folder="measured_example" ):
        rospy.init_node("record_detection")
        self.folder = "safety_labels"+folder
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(1200.0)) #tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        rospy.Subscriber("/pcl_gpu_tools/detected_objects", PoseArray, self.pose_cb)
        self.publisher = rospy.Publisher("/velodyne_points", PointCloud2)
        self.message_processed = True
        self.current_zone = 0
        self.index = 1
        self.distance = 1.0
        self.create_folder()
        time.sleep(2)

    def create_folder(self):
        #path = os.getcwd()
        try:
            os.mkdir(self.folder)
        except OSError:
            print ("Creation of the directory %s failed" % self.folder)
        else:
            print ("Successfully created the directory %s " % self.folder)


    def publish_gt(self, msg):
        self.publisher.publish(msg)
        self.message_processed = False

    def pose_cb(self,msg):
        stamp = msg.header.stamp.to_sec()
        frame_id = msg.header.frame_id
        default_class = "object"

        target_frame = "velodyne"

        #header = ",".join(["x", "y", "timestamp", "class", "frame_id"])
        #f.write(header + "\n")

        min_distance = 1000000000000000000
        safety_zone = -1

        for p in msg.poses:
            transform = self.tf_buffer.lookup_transform(target_frame,
                                           frame_id, #source frame
                                           rospy.Time(0), #get the tf at first available time
                                           rospy.Duration(0.1)) #wait for 1 second

            pose_stamped = PoseStamped()
            pose_stamped.header = msg.header
            pose_stamped.pose = p
            pose_transformed = tf2_geometry_msgs.do_transform_pose(pose_stamped, transform)
            p_dists = np.sqrt(np.power(pose_transformed.pose.position.x,2) +
                              np.power(pose_transformed.pose.position.y,2))
            #",".join([str(pose_transformed.pose.position.x), str(pose_transformed.pose.position.y)])
            #HACK
            if -2 < pose_transformed.pose.position.x < 0.7 and pose_transformed.pose.position.y < 0:
                if p_dists <= min_distance:

                    min_distance = p_dists
                    safety_zone = int(min_distance/self.distance)


        if  safety_zone != self.current_zone:
            print (safety_zone)
            f = open(os.path.join(self.folder, str(stamp)),"w")
            f.write(",".join([str(min_distance), str(safety_zone)])+"\n")
            f.close()
            f = open(os.path.join(self.folder,"events_recorded"),"a")
            f.write(str(stamp)+"\n")
            f.close()
            self.current_zone = safety_zone

        self.message_processed = True
        self.index +=1


if __name__ == '__main__':
    if __name__ == '__main__':
        parser = argparse.ArgumentParser(description = help_text)
        parser.add_argument("--bag", "-b", help="set input bagname")
        parser.add_argument("--group", "-g", default="nibio_2019")
        parser.add_argument("--topic", "-t", default="/velodyne_points")
        parser.add_argument("--debug", "-d", default=1)

        args = parser.parse_args()
        debug_mode = bool(int(args.debug))
        bag = rosbag.Bag(args.bag, mode="r")
        recoder = DetectionRecorder(folder=args.group)

        for topic, msg, t in bag.read_messages(topics=args.topic):
            recoder.publish_gt(msg)
            while not recoder.message_processed:
                #rospy.sleep(0.2)
                pass
            rospy.sleep(0.3)
        bag.close()
