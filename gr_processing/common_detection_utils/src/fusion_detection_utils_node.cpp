#include <common_detection_utils/common_detection_utils.h>

using namespace gr_detection;

int main(int argc, char **argv){
    std::cout << "PLOTTING EXITING IDS: "<<std::endl;
    FusionDetection* a = new FusionDetection();
    a->showCurrentDetections();


    std::shared_ptr<FusionDetection> foo;
    foo = std::make_shared<FusionDetection>();
    // cast of potentially incomplete object, but ok as a static cast:
    foo->showCurrentDetections();

    for (int i=0; i < 100; i++){
     for( auto it = foo->d_array_->DETECTIONSARRAY.begin(); it != foo->d_array_->DETECTIONSARRAY.end(); it++ ){
        std::cout << ":)";
    }
    }


}