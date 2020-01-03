#include <gr_fdd/recovery_executor.h>

using namespace gr_fdd;

RecoveryExecutor::RecoveryExecutor(YAML::Node node) {
    std::cout << "RE Constructor" << std::endl;
    for (YAML::Iterator a= node.begin(); a != node.end(); ++a){
        std::string name;
        a->first() >> name;//.as<std::string>();
        YAML::Node config;
        a->second() >> config;
        //strategy_selector_[name] = config["strategy"].as<std::string>();
        config["strategy"]>> strategy_selector_[name];
    }
}

std::string RecoveryExecutor::getRecoveryStrategy(std::string key){
    if (strategy_selector_[key].empty()){
        return "Unknown Error";
    }
    return strategy_selector_[key];    
}


RecoveryExecutor::~RecoveryExecutor() {
}

