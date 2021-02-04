#/usr/bin/python
from message_filters import ApproximateTimeSynchronizer, Subscriber
from darknet_ros_msgs.msg import CheckForObjectsAction, CheckForObjectsActionGoal, CheckForObjectsActionResult
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3Stamped
import rospy
import actionlib
import copy

class ImageSinAnimationLabeler(object):
    def __init__(self, depth = False):
        self.client = actionlib.SimpleActionClient('/darknet_ros/check_for_objects', CheckForObjectsAction)
        self.client.wait_for_server()
        rospy.loginfo("Darknet Server found")
        self.odom_frame = "odom"
        self.image = Image()
        self.id = 0
        self.person_pose = Vector3Stamped()
        self.is_processing = False
        image_sub = Subscriber("/camera/color/image_raw", Image, queue_size=10)
        depth_sub = Subscriber("/camera/depth/image_raw", Image, queue_size=10)
        pos_sub = Subscriber("/animated_human/location", Vector3Stamped, queue_size=10)

        self.depth = depth
        self.current_depth = None

        if depth:
            self.ats = ApproximateTimeSynchronizer([image_sub, depth_sub, pos_sub], queue_size=10, slop=1.0, allow_headerless=False)
            self.ats.registerCallback(self.depth_call)
        else:
            self.ats = ApproximateTimeSynchronizer([image_sub, pos_sub], queue_size=10, slop=1.0, allow_headerless=False)
            self.ats.registerCallback(self.call)

    def call(self, image, position):
        if self.is_processing:
            print "skip"
            return
        goal = CheckForObjectsActionGoal()
        goal.goal.id = self.id
        self.id = self.id + 1
        goal.goal.image = image
        self.image = image
        #print ("Called")
        self.person_pose = position
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(120.0))

    def depth_call(self, image, img_depth, position):
        if self.is_processing:
            print "skip"
            return
        goal = CheckForObjectsActionGoal()
        goal.goal.id = self.id
        self.id = self.id + 1
        goal.goal.image = image
        self.current_depth = img_depth
        self.image = image
        #print ("Called")
        self.person_pose = position
        self.client.send_goal(goal.goal,
                            active_cb=self.callback_active,
                            feedback_cb=self.callback_feedback,
                            done_cb=self.callback_done)
        self.client.wait_for_result(rospy.Duration.from_sec(120.0))


    def get_current_result(self):
        return self.client.get_result()

    def get_current_depth(self):
        return self.current_depth

    def callback_active(self):
        pass
        #rospy.loginfo("Goal has been sent to the action server.")

    def callback_done(self,state, result):
        #rospy.loginfo("Action server is done. State: %s, result: %s" % (str(state), str(result)))
        rospy.loginfo("This is the result state %d "% state)
        #if result:
            #print (self.client.get_result())

    def callback_feedback(self,feedback):
        rospy.loginfo("Feedback:%s" % str(feedback))

if __name__ == "__main__":
    rospy.init_node('image_sim_animation_label')
    imasin = ImageSinAnimationLabeler()
    rospy.spin()
