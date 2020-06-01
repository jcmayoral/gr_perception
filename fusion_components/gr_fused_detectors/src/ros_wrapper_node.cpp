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

    std::string path = ros::package::getPath("gr_fused_detectors");
    std::string config_path;
    std::string config_file;
    nh.param<std::string>("config_path", config_path, "config");
    nh.param<std::string>("config_file", config_file, "config.yaml");
    YAML::Node config_yaml = YAML::LoadFile((path+"/"+config_path+"/"+config_file).c_str());

    std::string name;
    std::string type;

    for(YAML::const_iterator it=config_yaml.begin();it!=config_yaml.end();++it) {
        std::cout << it->first << " name " << it->second << std::endl;
        name = it->first.as<std::string>();
        YAML::Node params(it->second);
        type = params["type"].as<std::string>();
        std::cout << "type " << params["type"] << std::endl;
        YAML::Node node_remappings(params["remappings"]);
        ros::M_string remappings = ros::names::getRemappings ();
        std::vector<std::string> argvs;   
        std::string r1, r2;
        for(YAML::const_iterator init=node_remappings.begin();init!=node_remappings.end();++init) {
            std::cout << init->first << " remaps to " << init->second << std::endl;
            r1 = init->first.as<std::string>().c_str();            
            r2 = init->second.as<std::string>().c_str();
            remappings.insert(std::pair<std::string,std::string>(r1,r2));
        }
        std::cout << "RESULT : "<< n.load(name, type, remappings, argvs) << std::endl;
    }
    ros::spin();
}
