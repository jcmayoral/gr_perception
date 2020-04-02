#include <geometry_msgs/Pose.h>

struct Person{
    int age;
};


struct PersonArray{
    std::map<int, Person> persons;
};