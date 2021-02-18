#/usr/bin/python
from message_filters import ApproximateTimeSynchronizer, Subscriber
from gr_action_msgs.msg import SimMotionPlannerAction, SimMotionPlannerActionGoal, SimMotionPlannerActionResult, SimMotionPlannerActionFeedback
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
import rospy
import actionlib
import random

class PersonSimAnimation(object):
    def __init__(self):
        self.pclient = actionlib.SimpleActionClient('/SimMotionPlanner/animated_human', SimMotionPlannerAction)
        self.pclient.wait_for_server()
        self.feedback = SimMotionPlannerActionFeedback()
        rospy.loginfo("Person Server found")

    def select_uniform_random(self, minvalue, maxvalue):
        return random.uniform(minvalue, maxvalue)

    def person_call(self):
        #rospy.loginfo("calling SimMotionPlanner")
        goal = SimMotionPlannerActionGoal()
        goal.goal.motion_type = "walk"
        goal.goal.setstart = True
        goal.goal.is_motion = True
        goal.goal.is_infinite_motion = False
        goal.goal.linearspeed = self.select_uniform_random(0.5,1.2)
        goal.goal.startpose.header.frame_id = "odom"
        goal.goal.startpose.pose.position.x = self.select_uniform_random(0.5,5)
        goal.goal.startpose.pose.position.y = self.select_uniform_random(-2,2)
        goal.goal.startpose.pose.orientation.w = 1.0

        goal.goal.goalPose.header.frame_id = "odom"
        goal.goal.goalPose.pose.position.x = self.select_uniform_random(7,10)
        goal.goal.goalPose.pose.position.y = self.select_uniform_random(-2,2)
        goal.goal.goalPose.pose.orientation.w = 1.0
        self.pclient.send_goal(goal.goal,
                            active_cb=self.pcallback_active,
                            feedback_cb=self.pcallback_feedback,
                            done_cb=self.pcallback_done)
        self.pclient.wait_for_result(rospy.Duration.from_sec(120.0))

    def pcallback_active(self):
        pass
        #rospy.loginfo("Goal has been sent to the action server.")

    def pcallback_done(self,state, result):
        pass
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        #rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print ()

    def pcallback_feedback(self,feedback):
        pass
        #rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    rospy.init_node('person_sim_animation_label')
    PersonSimAnimation().person_call()
    rospy.spin()
