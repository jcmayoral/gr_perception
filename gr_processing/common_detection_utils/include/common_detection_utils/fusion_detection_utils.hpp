#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <string>

namespace gr_detection{
    struct Person{
        int age = 10;
        geometry_msgs::Pose pose;
        geometry_msgs::Vector3 variance;
        int vari = 0;
        std::string id;
    };

    struct PersonArray{
        std::map<std::string, Person> persons;
    };

    //TODO add lockable class
    static PersonArray DETECTIONSARRAY;

    //FROM https://gist.github.com/abhijeetchopra/8e3068ef30702aeed84af0bb1fb87dd7
    //TODO HASH function
    std::string randomString(){
        std::string str = "AAAAAA";
        // string sequence
        str[0] = rand() % 26 + 65;
        str[1] = rand() % 26 + 65;
        str[2] = rand() % 26 + 65;

        // number sequence
        str[3] = rand() % 10 + 48;
        str[4] = rand() % 10 + 48;
        str[5] = rand() % 10 + 48;
        return str;
    }

    void UpdateOrientation(geometry_msgs::Quaternion q, std::string id){
        DETECTIONSARRAY.persons[id].pose.orientation = q;
    }

    void UpdateObject(std::string id, Person p){
        DETECTIONSARRAY.persons[id] = p;
        DETECTIONSARRAY.persons[id].age = 5;
    }

    void cleanUpCycle(){
        for( auto it = DETECTIONSARRAY.persons.begin(); it != DETECTIONSARRAY.persons.end();  ){
            if(it->second.age < 2){
                it = DETECTIONSARRAY.persons.erase(it);
                ROS_ERROR_STREAM("deleting person because has not been detected "<< DETECTIONSARRAY.persons.size()) ;
            }
            else{
                it->second.age--;
                it++;
            }
        }
    }

    void insertNewObject(Person p){
        p.id = "person_"+ gr_detection::randomString();
        p.age = 5;
        DETECTIONSARRAY.persons.insert(std::pair<std::string,Person>(p.id, p));   
    }

    Person GetObject(std::string id){
        return DETECTIONSARRAY.persons[id];
    }


    std::string matchDetection(Person new_cluster){
        std::vector<float> scores;
        std::vector<std::string> ids;

        for( auto it = DETECTIONSARRAY.persons.begin(); it != DETECTIONSARRAY.persons.end(); it++){
            float score = 0;
            score += std::abs(it->second.pose.position.x - new_cluster.pose.position.x);
            score += std::abs(it->second.pose.position.y - new_cluster.pose.position.y);
            score += std::abs(it->second.pose.position.z - new_cluster.pose.position.z);
            scores.push_back(score);
            ids.push_back(it->first);
        }

        if (scores.size()==0){
            return {};
        }

        auto min_score = min_element(scores.begin(), scores.end());
        //FIND Proper threshold
        //research minimum social distance
        if (*min_score < 1.0){
            int argmin = std::distance(scores.begin(), min_score);
            return ids[argmin];
        }

        return {};
    }


}