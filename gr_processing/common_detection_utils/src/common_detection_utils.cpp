#include "common_detection_utils/common_detection_utils.h"

using namespace gr_detection;

boost::shared_ptr<CustomArray> FusionDetection::d_array_(new CustomArray);

FusionDetection::FusionDetection(): time_break_{0.5}, minmatch_score_{2.0}{
    //DETECTIONSARRAY = new CustomArray();

}
FusionDetection::~FusionDetection(){

}

FusionDetection::FusionDetection(const FusionDetection& other){
    time_break_ = other.time_break_;
    minmatch_score_ = other.minmatch_score_;
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
    d_array_->DETECTIONSARRAY.at(id).last_update = clock();
    d_array_->DETECTIONSARRAY.at(id).pose.orientation = q;
}

void FusionDetection::UpdateObject(std::string id, Person p){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    d_array_->DETECTIONSARRAY.at(id) = p;
    d_array_->DETECTIONSARRAY.at(id).age = 5;
    d_array_->DETECTIONSARRAY.at(id).last_update = clock();
}

void FusionDetection::cleanUpCycle(){
    boost::mutex::scoped_lock lck(d_array_->mtx);
    clock_t curr_time = clock();

    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end();  ){
        //if(it->second.age < 2){
        double elapsed_seconds = double(curr_time - it->second.last_update) / CLOCKS_PER_SEC;

        if(elapsed_seconds>time_break_){
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
    p.id = "object_"+ randomString();
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

    if (d_array_->DETECTIONSARRAY.size() == 0){
        return NOPREVIOUSDETECTION;
    }

    for( auto it = d_array_->DETECTIONSARRAY.begin(); it != d_array_->DETECTIONSARRAY.end(); it++){
        /*
        float score = 1.0;
        score *= std::abs((it->second.pose.position.x - new_cluster.pose.position.x)/it->second.pose.position.x);
        score *= std::abs((it->second.pose.position.y - new_cluster.pose.position.y)/it->second.pose.position.y);
        score *= std::abs((it->second.variance.x - new_cluster.variance.x)/it->second.variance.x);
        score *= std::abs((it->second.variance.y - new_cluster.variance.y)/it->second.variance.y);
        score *= std::abs((it->second.variance.z - new_cluster.variance.z)/it->second.variance.z);
        //score += std::abs((it->second.volume - new_cluster.volume)/it->second.volume);
        */
        
        float score = 0;
        score += std::abs(it->second.pose.position.x - new_cluster.pose.position.x);
        score += std::abs(it->second.pose.position.y - new_cluster.pose.position.y);
        score += std::abs(it->second.pose.position.z - new_cluster.pose.position.z);

        score += std::abs(it->second.variance.x - new_cluster.variance.x);
        score += std::abs(it->second.variance.y - new_cluster.variance.y);
        score += std::abs(it->second.variance.z - new_cluster.variance.z);

        /*
        score += std::abs(it->second.speed.x - new_cluster.speed.x);
        score += std::abs(it->second.speed.y - new_cluster.speed.y);
        score += std::abs(it->second.speed.z - new_cluster.speed.z);
        */


        scores.push_back(score);
        ids.push_back(it->first);
    };
    }

    auto min_score = min_element(scores.begin(), scores.end());
    //FIND Proper threshold
    std::cout << "MIN SCORE " << *min_score << std::endl;

    if (*min_score < minmatch_score_){
        int argmin = std::distance(scores.begin(), min_score);
        return ids[argmin];
    }

    return {};
}
