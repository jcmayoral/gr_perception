#/usr/bin/python
from message_filters import ApproximateTimeSynchronizer, Subscriber
from gr_action_msgs.msg import SimMotionPlannerAction, SimMotionPlannerActionGoal, SimMotionPlannerActionResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
import rospy
import actionlib
import random

class PersonSimAnimation:
    def __init__(self):
        rospy.init_node('image_sim_animation_label')
        self.pclient = actionlib.SimpleActionClient('/SimMotionPlanner/animated_human', SimMotionPlannerAction)
        self.pclient.wait_for_server()
        rospy.loginfo("Server found")

    def select_uniform_random(self, minvalue, maxvalue):
        return random.uniform(minvalue, maxvalue)

    def person_call(self):
        goal = SimMotionPlannerActionGoal()
        goal.goal.motion_type = "walk"
        goal.goal.setstart = True
        goal.goal.is_motion = True
        goal.goal.is_infinite_motion = False
        goal.goal.linearspeed = 1.0
        goal.goal.startpose.pose.position.x = self.select_uniform_random(2,5)
        goal.goal.startpose.pose.orientation.w = 1.0

        goal.goal.goalPose.pose.position.x = self.select_uniform_random(8,12)
        goal.goal.goalPose.pose.orientation.w = 1.0
        print ("Calling new motion")
        self.pclient.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        #self.pclient.wait_for_result(rospy.Duration.from_sec(120.0))

    def callback_active(self):
        rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print (self.client.get_result())

    def callback_feedback(self,feedback):
        rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    PersonSimAnimation().person_call()
    rospy.spin()
