#! /usr/bin/python
import rospy
from image_sim_animation import ImageSinAnimationLabeler
from person_sim_animation import PersonSimAnimation

class SimAnimationManager(ImageSinAnimationLabeler, PersonSimAnimation):
    def __init__(self):
        #super(ImageSinAnimationLabeler, self).__init__()
        #super(PersonSimAnimation, self).__init__()
        self.count = 0
        PersonSimAnimation.__init__(self)
        ImageSinAnimationLabeler.__init__(self)
        self.backward_motion = False
        self.initialize = False

    def run(self):
        self.person_call()
        self.count = self.count + 1
        self.backward_motion = False
        self.initialize = False

    #Overridehas_turned
    def callback_feedback(self,feedback):
        rospy.loginfo("New Feedback:%s" % str(feedback))
        if not self.initialize:
            self.initialize = True
            return
        self.backward_motion =feedback.backward

    def callback_done(self,state, result):
        rospy.logwarn("new image bounding boxes for %s " %str(self.backward_motion))
        rospy.logwarn("person pose %s " %str(self.position))

        #print (self.image)

if __name__ == '__main__':
    rospy.init_node('image_sim_manager')
    manager = SimAnimationManager()

    for i in range(3):
        rospy.logerr("image request " + str(i) )
        manager.run()
    rospy.spin()
