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

    std::string path = ros::package::getPath("gr_fused_detectors");
    std::string config_path;
    std::string config_file;
    nh.param<std::string>("config_path", config_path, "config");
    nh.param<std::string>("config_file", config_file, "config.yaml");
    YAML::Node config_yaml = YAML::LoadFile((path+"/"+config_path+"/"+config_file).c_str());

    for(YAML::const_iterator it=config_yaml.begin();it!=config_yaml.end();++it) {
        //std::cout << it->first << " name " << it->second << std::endl;
        YAML::Node params(it->second);
        std::cout << params["type"] << "OK" << std::endl;
        YAML::Node remappings(params["remappings"]);
        for(YAML::const_iterator init=remappings.begin();init!=remappings.end();++init) {
            std::cout << init->first << " remaps to " << init->second << std::endl;
        }
    }

    std::string name = "/depth_nodelet";//ros::this_node::getNamespace();//ros::this;
    std::string type = "gr_depth_processing/MyNodeletClass";

    ros::M_string remappings = ros::names::getRemappings ();
    remappings.insert(std::pair<std::string,std::string>("color_info","/camera/color/camera_info"));
    remappings.insert(std::pair<std::string,std::string>("depth_info","/camera/depth/camera_info"));
    remappings.insert(std::pair<std::string,std::string>("color_frame","/camera/color/image_raw"));
    remappings.insert(std::pair<std::string,std::string>("depth_frame","/camera/depth/image_rect_raw"));
    remappings.insert(std::pair<std::string,std::string>("bounding_boxes","/darknet_ros/bounding_boxes"));

    std::vector<std::string> argvs;   
    std::cout << "RESULT : "<< n.load(name, type, remappings, argvs) << std::endl;

    auto name2="/depth_nodelet_2";
    remappings["color_info"] = "/camera2/color/camera_info";
    remappings["depth_info"] = "/camera2/depth/camera_info";
    remappings["color_frame"] = "/camera2/color/image_raw";
    remappings["depth_frame"] = "/camera2/depth/image_rect_raw";
    remappings["bounding_boxes"] = "/darknet_ros/bounding_boxes";
    
    std::cout << "RESULT : "<< n.load(name2, type, remappings, argvs) << std::endl;

    ros::spin();
}
