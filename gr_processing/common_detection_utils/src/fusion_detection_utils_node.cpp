#include <common_detection_utils/common_detection_utils.h>

using namespace gr_detection;

int main(int argc, char **argv){
    std::cout << "PLOTTING EXITING IDS: "<<std::endl;
    FusionDetection a;
    a.showCurrentDetections();
}