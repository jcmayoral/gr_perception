#include <ros/ros.h>
#include <common_detection_utils/common_detection_utils.h>
#include <geometry_msgs/PoseArray.h>

namespace gr_detection{
    class DetectorsFuser : gr_detection::FusionDetection{//:public gr_detection::FusionDetector {
        public:
        DetectorsFuser();
        ~DetectorsFuser();
        void timerCallback(const ros::TimerEvent& event);
        private:
        ros::Publisher rpub_;
        ros::Timer timer_;
        ros::NodeHandle nh_;
    };
}