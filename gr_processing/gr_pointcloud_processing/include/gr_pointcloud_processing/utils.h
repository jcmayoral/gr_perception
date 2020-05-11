#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>

struct Person{
    int age;
    geometry_msgs::Pose pose;
    geometry_msgs::Vector3 variance;
    int varid = 0; 
};

struct PersonArray{
    std::map<int, Person> persons;
};

int voting(PersonArray detected, Person new_cluster){
    std::vector<int> voting;
    std::vector<int> varids;
    //TODO UPDATE THIS TO SOME CONFIG FILE
    int comparisons_made = 7;

    int threshold = 5;
    float dist_threshold = 0.1;

    for(std::map<int, Person>::iterator it = detected.persons.begin(); it!=detected.persons.end(); it++ ) {
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
        if (std::abs(it->second.variance.x - new_cluster.variance.x)< threshold){
             positive_votes++;
        }
        if (std::abs(it->second.variance.y - new_cluster.variance.y)< threshold){
            positive_votes++;
        }
        if (std::abs(it->second.variance.z - new_cluster.variance.z)< threshold){
            positive_votes++;
        }

        if (std::abs(it->second.varid - new_cluster.varid)< threshold){
            positive_votes++;
        }
        std::cout << new_cluster.varid;
        voting.push_back(positive_votes);
        varids.push_back(new_cluster.varid);

    }
    std::cout << std::endl;

    if (voting.size()==0){
        std::cout << "any criterion is matched" << std::endl;
        return -1;
    }
    auto maxelement = max_element(voting.begin(), voting.end());
    std::cout << "Percentage accepted " << *maxelement/float(comparisons_made) << " positives "<< *maxelement << std::endl;

    if (*maxelement > 0){
        //get index with higher both
        int argmax = std::distance(voting.begin(), maxelement);
        //return id
        return varids[argmax];
    }

    return -1;
}

