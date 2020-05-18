#ifndef GRFUSIONPERSON_H
#define GRFUSIONPERSON_H

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <string>
#include <mutex>
#include <boost/thread/mutex.hpp>

namespace gr_detection{

    struct Person{
        int age = 10;
        geometry_msgs::Pose pose;
        geometry_msgs::Vector3 variance;
        int vari = 0;
        std::string id;
    };

    typedef std::map<std::string, Person> CustomMap;

    class CustomArray{
        public:
            CustomArray(): mtx(){

            };
            ~CustomArray(){

            };
            CustomMap DETECTIONSARRAY;
            boost::mutex mtx;
    };

    class FusionDetection{
        public:
            FusionDetection();
            ~FusionDetection();
            FusionDetection(const FusionDetection& other);
            int getDetectionsNumber();
            std::string randomString();
            void UpdateOrientation(geometry_msgs::Quaternion q, std::string id);
            void UpdateObject(std::string id, Person p);
            void cleanUpCycle();
            void showCurrentDetections();
            void insertNewObject(Person p);
            Person GetObject(std::string id);
            std::string matchDetection(Person new_cluster);
            static CustomArray* d_array_;
    };
};
#endif