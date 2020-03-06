#include<gr_thermal_processing/thermal_processing.h>
#include <ros/ros.h>
using namespace gr_thermal_processing;

int main(int argc, char** argv){
    ros::init(argc,argv, "thermal_camera_node");
    ThermalProcessing();
}