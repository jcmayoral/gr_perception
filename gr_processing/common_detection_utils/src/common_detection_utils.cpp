#include "common_detection_utils/common_detection_utils.h"

using namespace gr_detection;

DetectionArray FusionDetection::DETECTIONSARRAY = DetectionArray();


FusionDetection::FusionDetection(){

}
FusionDetection::~FusionDetection(){

}

FusionDetection::FusionDetection(const FusionDetection& other){
    std::cout << "calling copy constructor"<< std::endl;
}

int FusionDetection::getDetectionsNumber(){
    return DETECTIONSARRAY.size();
}

std::string FusionDetection::randomString(){
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

void FusionDetection::UpdateOrientation(geometry_msgs::Quaternion q, std::string id){
    DETECTIONSARRAY[id].pose.orientation = q;
}

void FusionDetection::UpdateObject(std::string id, Person p){
    DETECTIONSARRAY[id] = p;
    DETECTIONSARRAY[id].age = 5;
}

void FusionDetection::cleanUpCycle(){
    for( auto it = DETECTIONSARRAY.begin(); it != DETECTIONSARRAY.end();  ){
        if(it->second.age < 2){
            it = DETECTIONSARRAY.erase(it);
        }
        else{
            it->second.age--;
            it++;
        }
    }
}

void FusionDetection::showCurrentDetections(){
    std::cout << getDetectionsNumber() << std::endl;
    for( auto it = DETECTIONSARRAY.begin(); it != DETECTIONSARRAY.end(); it++){
        std::cout << it->first << std::endl;
    }
}

void FusionDetection::insertNewObject(Person p){
    p.id = "person_"+ randomString();
    p.age = 5;
    DETECTIONSARRAY.insert(std::pair<std::string,Person>(p.id, p));   
}

Person FusionDetection::GetObject(std::string id){
    return DETECTIONSARRAY[id];
}

std::string FusionDetection::matchDetection(Person new_cluster){
    std::vector<float> scores;
    std::vector<std::string> ids;

    for( auto it = DETECTIONSARRAY.begin(); it != DETECTIONSARRAY.end(); it++){
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