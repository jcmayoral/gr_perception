#include <gr_fused_detectors/fused_detectors.h>

using namespace gr_detection;

DetectorsFuser::DetectorsFuser(): nh_{}{
    rpub_ = nh_.advertise<geometry_msgs::PoseArray>("fused_detectors", 1, true);
    timer_ = nh_.createTimer(ros::Duration(0.1), &DetectorsFuser::timerCallback,this);
}

void DetectorsFuser::timerCallback(const ros::TimerEvent& event){
    std::cout << "TIMER " << std::endl;
    rpub_.publish(createPoseArray());
}

DetectorsFuser::~DetectorsFuser(){

}