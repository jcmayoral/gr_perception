#ifndef GRFUSIONPERSON_H
#define GRFUSIONPERSON_H

#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <string>
#include <mutex>
#include <ctime>
#include <boost/thread/mutex.hpp>
#include <geometry_msgs/PoseArray.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>


namespace gr_detection{
    const std::string NOPREVIOUSDETECTION="ND";
    const float OUT_OF_RANGE=-1.0;

    struct Person{
        int age = 10;
        clock_t last_update = clock();
        geometry_msgs::Pose pose;
        geometry_msgs::Vector3 variance;
        geometry_msgs::Vector3 speed;
        int vari = 0;
        std::string id;
        float volume;
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
            void UpdateOrientation(std::string id, Person p);
            void UpdateObject(std::string id, Person p);
            void cleanUpCycle();
            void showCurrentDetections();
            void insertNewObject(Person p);
            Person GetObject(std::string id);
            std::string matchDetection(Person new_cluster);
            geometry_msgs::PoseArray createPoseArray();
            geometry_msgs::Quaternion getDifferenceQuaternion(geometry_msgs::Quaternion a, geometry_msgs::Quaternion b);
            
            static boost::shared_ptr<CustomArray> d_array_;
            float time_break_;
            float minmatch_score_;
    };
};
#endif