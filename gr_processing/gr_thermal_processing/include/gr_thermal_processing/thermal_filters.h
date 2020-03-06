#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>

#include <math.h>
#include <random>

void cv_filter(cv::Mat& frame){
    try{
        cv::GaussianBlur(frame, frame, cv::Size(5,5), 1, 0, cv::BORDER_DEFAULT);
        //im.at<uint16_t>(cell_x+cell_y*im.rows);
        //cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY );
        int erosion_size = 2.0;
        cv::Mat element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                        cv::Size( 3*erosion_size + 1, 3*erosion_size+1 ),
                                        cv::Point( erosion_size, erosion_size ) );
        cv::erode(frame, frame, element);
        cv::dilate(frame, frame, element);
        //output_frame = detectPeople(input_frame);
        cv::Laplacian(frame, frame, CV_16U);
        //cv::Canny(frame, frame,120, 200 );
      }
      catch( cv::Exception& e ){
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