#include <common_detection_utils/common_detection_utils.h>

using namespace gr_detection;


int main(int argc, char **argv){
    std::cout << "PLOTTING EXITING IDS: "<<std::endl;
    FusionDetection* a = new FusionDetection();
    //boost::shared_ptr<CustomArray> b;
    //b = boost::static_pointer_cast<CustomArray>(a->d_array_);
    a->showCurrentDetections();
    //a->showCurrentDetections();
    std::cout << "DN " <<a->getDetectionsNumber()<<std::endl;

    /*
    b->mtx.lock();
    for( auto it = b->DETECTIONSARRAY.begin(); it != b->DETECTIONSARRAY.end(); it++){
        std::cout << "with  b "<< it->first << std::endl;
    }
    b->mtx.unlock();
    */
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