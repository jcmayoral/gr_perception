#include <geometry_msgs/Pose.h>

struct Person{
    int age;
    float size_x;
    float size_y;
    float size_z;
    geometry_msgs::Pose pose;
};


struct PersonArray{
    double age;
    std::map<std::string, Person> persons;
};