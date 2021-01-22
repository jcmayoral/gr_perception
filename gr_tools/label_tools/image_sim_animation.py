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
        image_sub = Subscriber("/camera/color/image_raw", Image, queue_size=10)
        pos_sub = Subscriber("/animated_human/location", Vector3Stamped, queue_size=10)
        self.ats = ApproximateTimeSynchronizer([image_sub, pos_sub], queue_size=10, slop=1.0, allow_headerless=False)
        self.ats.registerCallback(self.call)
        rospy.spin()

    def call(self, image, position):
        goal = CheckForObjectsActionGoal()
        goal.goal.id = 1
        goal.goal.image = image
        print ("Called")
        self.client.send_goal(goal.goal)
        result = self.client.wait_for_result(rospy.Duration.from_sec(5.0))
        print("This is the result", result)
        if result:
            print (self.client.get_result()  )


if __name__ == "__main__":
    ImageSinAnimationLabeler()
