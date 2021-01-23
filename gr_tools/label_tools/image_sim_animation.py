#/usr/bin/python
from message_filters import ApproximateTimeSynchronizer, Subscriber
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
import rospy
import actionlib

class ImageSinAnimationLabeler:
    def __init__(self):
        rospy.init_node('image_sim_animation_label')
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        rospy.loginfo("Server found")
        self.image = Image()
        image_sub = Subscriber("/camera/color/image_raw", Image, queue_size=10)
        pos_sub = Subscriber("/animated_human/location", Vector3Stamped, queue_size=10)
        self.ats = ApproximateTimeSynchronizer([image_sub, pos_sub], queue_size=10, slop=1.0, allow_headerless=False)
        self.ats.registerCallback(self.call)
        rospy.spin()

    def call(self, image, position):
        goal = CheckForObjectsActionGoal()
        goal.goal.id = 1
        goal.goal.image = image
        self.image = image
        print ("Called")
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(120.0))

    def callback_active(self):
        rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print (self.client.get_result())

    def callback_feedback(feedback):
        rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    ImageSinAnimationLabeler()
