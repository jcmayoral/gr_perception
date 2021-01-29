#/usr/bin/python
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import PointCloud2
import rospy
import actionlib

FILEPATH = "/media/jose/ADATA HD710/PHD/datasets/FIELDSAFE/filter_bag_0.bag"
PC_PATH = "/home/jose/ros_ws/src/gr_perception/gr_tools/label_tools/fieldsafe_filter/pc_testdataset/"

class ImageFieldSafeLabeler:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        self.id = 0
        self.result = CheckForObjectsActionResult()
        rospy.loginfo("PC Server found")

    def call(self, img):
        goal = CheckForObjectsActionGoal()
        self.id = self.id + 1
        goal.goal.image = img
        #goal.goal.image = image
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(120.0))
        return self.client.get_result()

    def callback_active(self):
        rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        pass
        #rospy.loginfo("This is the result state %d "% state)
        #rospy.loginfo("This is the result result %s "% str(result))

    def callback_feedback(self,feedback):
        rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    rospy.init_node('pc_fieldsafe_label')
    import os, sys
    try:
        os.mkdir("image_testdataset")
    except:
        pass
        #sys.exit()
    try:
        os.chdir("image_testdataset")
    except:
        sys.exit()


    import rosbag
    bag = rosbag.Bag(FILEPATH)
    imglabeler = ImageFieldSafeLabeler()
    filename = ""
    count = 0

    import numpy as np
    all_pcfiles = os.listdir(PC_PATH)
    all_timestamps = [np.double(d)/1000000000 for d in all_pcfiles]

    for topic, msg, t in bag.read_messages(topics=['/Multisense/left/image_rect_color']):
        #rospy.sleep(0.05)
        filename = str(msg.header.stamp.to_nsec())
        current_result = imglabeler.call(msg)
        with open(filename+".txt",'a') as f:
            if len(current_result.bounding_boxes.bounding_boxes)< 1:
                print "NO Object detected"
                continue

            for detection in current_result.bounding_boxes.bounding_boxes:
                if detection.Class != "person":
                    print "Ignoring " , detection.Class
                    continue
                else:
                    print "person found to be implemented"
                    count += 1
                    #continue

                nfilename = np.double(filename)/1000000000
                arg_min = np.argmin([np.fabs(nfilename-d) for d in all_timestamps])
                print "closer pointcloud ", all_pcfiles[arg_min]
                print "Img " , nfilename
                print "closest ", np.min([np.fabs(nfilename-d) for d in all_timestamps])
                continue


                if detection.pose.position.x > 0:
                    #x: -0.155390086843 FILTER the fucking car that it's detected all frmes
                    f.write("%f %f %f "%(detection.pose.position.x, detection.pose.position.y, detection.pose.position.z))
                    f.write("%f %f %f %f "%(detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z,  detection.pose.orientation.w))
                    f.write("%f %f %f "%(detection.speed.x, detection.speed.y, detection.speed.z))
                    f.write("%f \n"%(detection.is_dynamic))

    bag.close()
    print "final count : ", count
    #rospy.spin()
