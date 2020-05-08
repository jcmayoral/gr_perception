#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Vector3.h>

struct Person{
    int age;
    geometry_msgs::Pose pose;
    geometry_msgs::Vector3 variance;
    int varid; 
};

struct PersonArray{
    std::map<int, Person> persons;
};

int voting(PersonArray detected, Person new_cluster){
    std::vector<int> voting;
    std::vector<int> varids;

    int threshold = 10;

    for(std::map<int, Person>::iterator it = detected.persons.begin(); it!=detected.persons.end(); it++ ) {
        int positive_votes = 0;
        /*
        if (fabs(*it.pose.x - new_cluster.pose.x)< threshold{
            positive_votes++;
        }
        if (fabs(*it.pose.y - new_cluster.pose.y)< threshold{
            positive_votes` ++;
        }
        if (fabs(*it.pose.z - new_cluster.pose.z)< threshold{
            positive_votes++;
        }
        if (fabs(*it.variance.x - new_cluster.variance.x)< threshold{
             positive_votes++;
        }
        if (fabs(*it.variance.y - new_cluster.variance.y)< threshold{
            positive_votes++;
        }
        if (fabs(*it.variance.z - new_cluster.posvariance.z)< threshold){
            positive_votes++;
        }

        */
        if (std::abs(it->second.varid - new_cluster.varid)< threshold){
            positive_votes++;
        }
        voting.push_back(positive_votes);
        varids.push_back(new_cluster.varid);

    }

    auto maxelement = max_element(voting.begin(), voting.end());
    std::cout << "MAX ELEMENT " << *maxelement << std::endl;

    if (*maxelement > 0){
        //get index with higher both
        int argmax = std::distance(voting.begin(), maxelement);
        //return id
        return varids[argmax];
    }

    return -1;
}

