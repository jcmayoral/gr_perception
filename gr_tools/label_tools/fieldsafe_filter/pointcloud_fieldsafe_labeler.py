#/usr/bin/python
from gr_action_msgs.msg import GRPCProcessAction, GRPCProcessActionGoal, GRPCProcessActionResult
from sensor_msgs.msg import PointCloud2
import rospy
import actionlib

FILEPATH = "/media/jose/ADATA HD710/PHD/datasets/FIELDSAFE/pointcloud_example_2021-01-29-17-13-28.bag"

class PCFieldSafeLabeler:
    def __init__(self):
        self.client = actionlib.SimpleActionClient('/pointcloud_lidar_processing/gr_pointcloud/process', GRPCProcessAction)
        self.client.wait_for_server()
        self.id = 0
        self.result = GRPCProcessActionResult()
        rospy.loginfo("PC Server found")

    def call(self, pc):
        goal = GRPCProcessActionGoal()
        self.id = self.id + 1
        goal.goal.goal_pc = pc
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
        os.mkdir("testdataset")
    except:
        pass
        #sys.exit()
    try:
        os.chdir("testdataset")
    except:
        sys.exit()


    import rosbag
    bag = rosbag.Bag(FILEPATH)
    pclabeler = PCFieldSafeLabeler()
    filename = ""
    for topic, msg, t in bag.read_messages(topics=['/velodyne_points']):
        #rospy.sleep(0.05)
        filename = str(msg.header.stamp.to_nsec())
        current_result = pclabeler.call(msg)
        with open(filename,'a') as f:
            if len(current_result.found_objects.objects)> 0:
                print(current_result)

            for detection in current_result.found_objects.objects:
                if detection.pose.position.x > 0:
                    #x: -0.155390086843 FILTER the fucking car that it's detected all frmes
                    f.write("%f %f %f "%(detection.pose.position.x, detection.pose.position.y, detection.pose.position.z))
                    f.write("%f %f %f %f "%(detection.pose.orientation.x, detection.pose.orientation.y, detection.pose.orientation.z,  detection.pose.orientation.w))
                    f.write("%f %f %f "%(detection.speed.x, detection.speed.y, detection.speed.z))
                    f.write("%f \n"%(detection.is_dynamic))

    bag.close()
    #rospy.spin()
