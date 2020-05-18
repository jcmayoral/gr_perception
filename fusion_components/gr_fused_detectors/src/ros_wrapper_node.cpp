#include "nodelet/nodelet.h"
#include "nodelet/loader.h"
#include <ros/ros.h>
#include <gr_fused_detectors/fused_detectors.h>

using namespace gr_detection;

int main(int argc, char** argv){
    ros::init(argc, argv, "aaaa");
    DetectorsFuser detectors_fuser;

    nodelet::Loader n(false);
    ros::NodeHandle nh;
    nodelet::Loader n2(false);

    ROS_ERROR("WIP...");
        ros::Duration(10).sleep();

    std::string name = "gr_depth_processing/MyNodeletClass";
    std::string type = "gr_depth_processing/MyNodeletClass";

    ros::M_string remappings = ros::names::getRemappings ();
    std::vector<std::string> argvs;

    
    auto name2="gr_pointcloud_processing/PointCloudProcessor";
    auto type2="gr_pointcloud_processing/PointCloudProcessor";
    
    std::cout << "RESULT : "<< n2.load(name2, type2, remappings, argvs) << std::endl;
    std::cout << "RESULT : "<< n.load(name, type, remappings, argvs) << std::endl;

    /*
    FusionDetection* a = new FusionDetection();
    Person p1;
    FusionDetection* b = new FusionDetection();
    std::cout << "insert  A "<<std::endl;
    a->insertNewObject(p1);
    std::cout << "show B"<<std::endl;
    b->showCurrentDetections();
    Person p2;
    std::cout << "insert B"<<std::endl;
    b->insertNewObject(p2);
    std::cout << "show A"<<std::endl;
    a->showCurrentDetections();
    */
    ros::spin();
}
