#include "common_detection_utils/common_detection_utils.h"

using namespace gr_detection;

CustomArray* FusionDetection::d_array_ = new CustomArray();

FusionDetection::FusionDetection(){
    std::cout << "constructor ";
    //DETECTIONSARRAY = new CustomArray();

}
FusionDetection::~FusionDetection(){

}

FusionDetection::FusionDetection(const FusionDetection& other){
    std::cout << "calling copy constructor"<< std::endl;
}

int FusionDetection::getDetectionsNumber(){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    return d_array_->DETECTIONSARRAY.size();
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
    boost::mutex::scoped_lock lck(d_array_->mtx);
    d_array_->DETECTIONSARRAY.at(id).pose.orientation = q;
}

void FusionDetection::UpdateObject(std::string id, Person p){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    d_array_->DETECTIONSARRAY.at(id) = p;
    d_array_->DETECTIONSARRAY.at(id).age = 5;
}

void FusionDetection::cleanUpCycle(){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end();  ){
        if(it->second.age < 2){
            it = d_array_->DETECTIONSARRAY.erase(it);
        }
        else{
            it->second.age--;
            it++;
        }
    }
}

void FusionDetection::showCurrentDetections(){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end(); it++){
        std::cout << it->first << std::endl;
    }
}

geometry_msgs::PoseArray FusionDetection::createPoseArray(){
    geometry_msgs::PoseArray array;
    //TODO SET THIS LINK SOMEWHERE
    array.header.frame_id = "base_link";
    boost::mutex::scoped_lock lck(d_array_->mtx);

    geometry_msgs::Pose p;
    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end(); it++){
        p = it->second.pose;
        array.poses.push_back(p);
    }
    return array;
}

void FusionDetection::insertNewObject(Person p){
    p.id = "person_"+ randomString();
    p.age = 5;    
    boost::mutex::scoped_lock lck(d_array_->mtx);{
    d_array_->DETECTIONSARRAY.insert(std::pair<std::string,Person>(p.id, p));
    }
}

Person FusionDetection::GetObject(std::string id){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    return d_array_->DETECTIONSARRAY.at(id);
}

std::string FusionDetection::matchDetection(Person new_cluster){
    std::vector<float> scores;
    std::vector<std::string> ids;

    boost::mutex::scoped_lock lck(d_array_->mtx);{
    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end(); it++){
        float score = 0;
        score += std::abs(it->second.pose.position.x - new_cluster.pose.position.x);
        score += std::abs(it->second.pose.position.y - new_cluster.pose.position.y);
        score += std::abs(it->second.pose.position.z - new_cluster.pose.position.z);
        scores.push_back(score);
        ids.push_back(it->first);
    };
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
