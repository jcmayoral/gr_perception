#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>
#include <string>

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

std::string voting(PersonArray detected, Person new_cluster){
    std::vector<int> voting;
    std::vector<int> varintensities;
    std::vector<std::string> ids;
    //TODO UPDATE THIS TO SOME CONFIG FILE
    int comparisons_made = 2;
    int threshold = 5;
    float dist_threshold = 0.1;

    for( auto it = detected.persons.begin(); it != detected.persons.end(); it++){
        int positive_votes = 0;
        if (std::abs(it->second.pose.position.x - new_cluster.pose.position.x)< dist_threshold){
            positive_votes++;
        }

        if (std::abs(it->second.pose.position.y - new_cluster.pose.position.y)< dist_threshold){
            positive_votes++;
        }

        if (std::abs(it->second.pose.position.z - new_cluster.pose.position.z)< dist_threshold){
            positive_votes++;
        }
        /*
        if (std::abs(it->second.variance.x - new_cluster.variance.x)< threshold){
             positive_votes++;
        }
        if (std::abs(it->second.variance.y - new_cluster.variance.y)< threshold){
            positive_votes++;
        }
        if (std::abs(it->second.variance.z - new_cluster.variance.z)< threshold){
            positive_votes++;
        }
        if (std::abs(it->second.vari - new_cluster.vari)< threshold){
            positive_votes++;
        }
        */
        voting.push_back(positive_votes);
        varintensities.push_back(new_cluster.vari);
        ids.push_back(it->first);
    }

    if (voting.size()==0){
        return {};
    }
    auto maxelement = max_element(voting.begin(), voting.end());
    //std::cout << "Percentage accepted " << *maxelement/float(comparisons_made) << " positives "<< *maxelement << std::endl;

    if (*maxelement > 0){
        //get index with higher both
        int argmax = std::distance(voting.begin(), maxelement);
        //return id
        return ids[argmax];
    }

    return {};
}



std::string scoreFunction(PersonArray detected, Person new_cluster){
    std::vector<float> scores;
    std::vector<int> varintensities;
    std::vector<std::string> ids;

    for( auto it = detected.persons.begin(); it != detected.persons.end(); it++){
        float score = 0;
        score += std::abs(it->second.pose.position.x - new_cluster.pose.position.x);
        score += std::abs(it->second.pose.position.y - new_cluster.pose.position.y);
        score += std::abs(it->second.pose.position.z - new_cluster.pose.position.z);
        std::cout << "SC: " << score << std::endl;
        scores.push_back(score);
        varintensities.push_back(new_cluster.vari);
        ids.push_back(it->first);
    }

    if (scores.size()==0){
        return {};
    }

    auto min_score = min_element(scores.begin(), scores.end());
    std::cout << "MIN SCORE " << *min_score << std::endl;
    //FIND Proper threshold
    if (*min_score < 1.0){
        int argmin = std::distance(scores.begin(), min_score);
        return ids[argmin];
    }

    return {};
}
