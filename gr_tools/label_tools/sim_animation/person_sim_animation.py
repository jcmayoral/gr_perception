#/usr/bin/python
from message_filters import ApproximateTimeSynchronizer, Subscriber
from gr_action_msgs.msg import SimMotionPlannerAction, SimMotionPlannerActionGoal, SimMotionPlannerActionResult, SimMotionPlannerActionFeedback
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
import rospy
import actionlib
import random

class PersonSimAnimation(object):
    def __init__(self, class_id):
        self.class_id = class_id
        self.min_x = 0.5 + (class_id * 2.0)
        self.max_x = 1.5 + (class_id * 2.0)
        if class_id == 3:
            self.max_x = 12.0
        self.pclient = actionlib.SimpleActionClient('/SimMotionPlanner/animated_human', SimMotionPlannerAction)
        self.pclient.wait_for_server()
        self.feedback = SimMotionPlannerActionFeedback()
        rospy.loginfo("Person Server found")

    def select_uniform_random(self, minvalue, maxvalue):
        return random.uniform(minvalue, maxvalue)

    def person_call(self):
        #motion_types = ["walk", "stand_up", "sit_down", "stand_up", "talk_a", "talk_b"]
        motion_types = ["walk", "moonwalk", "talk_a", "talk_b"]
        #rospy.loginfo("calling SimMotionPlanner")
        goal = SimMotionPlannerActionGoal()
        goal.goal.motion_type = motion_types[random.randint(0, 3)]
        goal.goal.setstart = True
        goal.goal.is_motion = True
        goal.goal.is_infinite_motion = False

        goal.goal.linearspeed = self.select_uniform_random(0.2,0.7)
        zoffset = self.select_uniform_random(-0.10,0.10)
        goal.goal.startpose.header.frame_id = "odom"
        goal.goal.startpose.pose.position.x = self.select_uniform_random(self.min_x,self.max_x)
        goal.goal.startpose.pose.position.y = self.select_uniform_random(-2,0)
        goal.goal.startpose.pose.position.z = -zoffset
        goal.goal.startpose.pose.orientation.w = 1.0

        goal.goal.goalPose.header.frame_id = "odom"
        goal.goal.goalPose.pose.position.x = self.select_uniform_random(self.min_x,self.max_x)
        goal.goal.goalPose.pose.position.y = self.select_uniform_random(0,2)
        goal.goal.goalPose.pose.position.z = -zoffset
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
