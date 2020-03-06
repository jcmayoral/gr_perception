#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <cv_bridge/cv_bridge.h>

#include <math.h>
#include <random>

void cv_filter(cv_bridge::CvImagePtr& frame){
    try{
        //cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_CONSTANT);
        //im.at<uint16_t>(cell_x+cell_y*im.rows);
        //cv::cvtColor(frame->image, frame->image, cv::COLOR_BGR2GRAY );
        //int erosion_size = 2.0;
        //cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
          //                              cv::Size( 3*erosion_size + 1, 3*erosion_size+1 ),
            //                            cv::Point( erosion_size, erosion_size ) );
        //cv::erode(frame->image, frame->image, element);
        //cv::dilate(frame->image, frame->image, element);

        //Original image comin on RGB no idea why
       //   Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
       cv::Laplacian(frame->image, frame->image, CV_8UC1, 3, 1,0, cv::BORDER_CONSTANT);
       //cv::cvtColor(frame->image, frame->image, cv::COLOR_BGR2GRAY );
       cv::convertScaleAbs(frame->image, frame->image);

      }
      catch( cv::Exception& e ){
        std::cout << "GOING WRONG" << std::endl;
        return;
      }
}

double get_variance(float mean, uint16_t* points, int points_number, double* score_array){
    double var = 0;
    double scoring;

    for(int n = 0; n< points_number; ++n){
        scoring = points[n];//pow(points[n] - mean,2);
        score_array[n] = scoring;
        var += scoring;
    }
    var /= points_number;
    return sqrt(var);
}