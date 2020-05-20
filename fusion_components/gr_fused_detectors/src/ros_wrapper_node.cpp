#include "nodelet/nodelet.h"
#include "nodelet/loader.h"
#include <ros/ros.h>
#include <gr_fused_detectors/fused_detectors.h>

using namespace gr_detection;

int main(int argc, char** argv){
    ros::init(argc, argv, "aaaa");
    DetectorsFuser detectors_fuser;
    ros::NodeHandle nh("~");
    nodelet::Loader n(false);
    std::cout << nh.getNamespace() << std::endl;

    std::string name = "/depth_nodelet";//ros::this_node::getNamespace();//ros::this;
    std::string type = "gr_depth_processing/MyNodeletClass";

    ros::M_string remappings = ros::names::getRemappings ();
    remappings.insert(std::pair<std::string,std::string>("color_info","/color_info1"));
    remappings.insert(std::pair<std::string,std::string>("depth_info","/depth_info1"));
    remappings.insert(std::pair<std::string,std::string>("color_frame","/depth_frame1"));
    remappings.insert(std::pair<std::string,std::string>("depth_frame","/depth_frame1"));
    remappings.insert(std::pair<std::string,std::string>("bounding_boxes","/bounding_boxes1"));

    std::vector<std::string> argvs;   
    std::cout << "RESULT : "<< n.load(name, type, remappings, argvs) << std::endl;

    auto name2="/depth_nodelet_2";
    remappings["color_info"] = "/color_info2";
    remappings["depth_info"] = "/depth_info2";
    remappings["color_frame"] = "/depth_frame2";
    remappings["depth_frame"] = "/depth_frame2";
    remappings["bounding_boxes"] = "/bounding_boxes2";
    
    std::cout << "RESULT : "<< n.load(name2, type, remappings, argvs) << std::endl;

    ros::spin();
}
