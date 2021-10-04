#/usr/bin/python
from gr_action_msgs.msg import GRPCProcessAction, GRPCProcessActionGoal, GRPCProcessActionResult
from sensor_msgs.msg import PointCloud2
import rospy
import actionlib

FILEPATH = "/media/jose/ADATA HD710/PHD/datasets/FIELDSAFE/pointcloud_example_2021-01-29-17-13-28.bag"

class PCProcessingClient:
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
