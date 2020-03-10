#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <geometry_msgs/Accel.h>
#include <cv_bridge/cv_bridge.h>

#include <math.h>
#include <random>

void cv_filter(cv_bridge::CvImagePtr& frame, geometry_msgs::Accel& output){
    try{
        cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
        //im.at<uint16_t>(cell_x+cell_y*im.rows);
        //cv::cvtColor(frame->image, frame->image, cv::COLOR_BGR2GRAY );
        int erosion_size = 1.0;
        cv::Mat element; 
        //cv::erode(frame->image, frame->image, element);
        //cv::dilate(frame->image, frame->image, element);

        //Original image comin on RGB no idea why
       //   Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
       cv::Mat l1;
       cv::Mat l2;
       cv::Mat kernel;
       cv::Point anchor( -1, -1 );
       double delta;
       int ddepth = CV_16U;
       int kernel_size;
       delta = 0;
       ddepth = 0;
      
       //cv::Laplacian(frame->image, frame->image, CV_8UC1, 3, 3,0);
       //cv::Laplacian(l1, l2, CV_8UC1, 3, 3,0);
       //cv::Laplacian(l1, frame->image, CV_8UC1, 3, 3,0);
       //cv::GaussianBlur(frame->image, l1, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
       //cv::Laplacian(l1, l1, CV_8UC1, 5, 1,0, cv::BORDER_DEFAULT);
       //cv::GaussianBlur(l1, l2, cv::Size(5,5), 1, 0, cv::BORDER_DEFAULT);
       //cv::Laplacian(l2, l2, CV_8UC1, 5, 1,0, cv::BORDER_DEFAULT);
       //frame->image = l2 + l1;
       //cv::cvtColor(frame->image, frame->image, cv::COLOR_BGR2GRAY );
       //cv::convertScaleAbs(frame->image, frame->image);

       cv::Mat myKernel((cv::Mat_< uchar >(5, 5) <<
                                 .1, 0,  -.1,  0, .1,
                                  0, .1,  -.2,  .1, 0,
                                 -.1, -.2, 1.8, -.2, -.1,
                                  0, .1,  -.2,  .1, 0,
                                 .1, 0,  -.1,  0, .1));
       for (int i =0; i<7; i++){
           erosion_size = 1.0*(i+1);
           element = cv::getStructuringElement( cv::MORPH_ELLIPSE,
                                        cv::Size( erosion_size + 3, erosion_size+3 ),
                                        cv::Point( erosion_size, erosion_size ) );
           cv::erode(frame->image, frame->image, element);
           //cv::GaussianBlur(frame->image, frame->image, cv::Size(7,7), 1, 0, cv::BORDER_DEFAULT);
           kernel_size = 7 - i ;
            kernel = cv::Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
           cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
           cv::dilate(frame->image, frame->image, element);
       }
       //cv::dilate(frame->image, frame->image, element);
       //cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
       //cv::erode(frame->image, frame->image, element);
       //
       
       //cv::bilateralFilter(frame->image, l1, 5, 50.0, 50.0);//, cv::BORDER_DEFAULT);
       //frame->image = l1;

       
       
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::Laplacian(frame->image, frame->image, CV_8UC1, 3, 3,0);
      cv::convertScaleAbs(frame->image, frame->image);
       cv::Scalar mu, sigma;
       cv::meanStdDev(frame->image, mu, sigma);
       //std::cout << mu << " , " << sigma << std::endl;

      // cv::Canny(frame->image, l1, 0,10, 7);
       //cv::Mat dst;
       //dst = cv::Scalar::all(0);
       //frame->image.copyTo(dstz, l1);
       //frame->image = dst;

       //three channels mean is practically same somehow
       //normalizing
       mu[0] /= 255;
       sigma[0]/=255;

       output.linear.x = mu[0];
       output.angular.x = sigma[0];

       output.linear.y = exp(output.linear.x);
       output.angular.y = exp(output.angular.x);

       output.angular.z = exp(0.1/ output.angular.x);//output.angular.z/log(sigma[2]) + 0.1;
       //std::cout << output.linear.x * output.linear.y * output.linear.z * output.angular.x * output.angular.y * output.angular.z << std::endl;
       //std::cout << output.linear.x * output.linear.y * output.angular.x * output.angular.y * output.angular.z << std::endl;
       //output.linear.z =  log(output.angular.y * output.linear.y);///log(mu[2]) + 0.1;
       output.linear.z = exp(0.4 / output.linear.x) ;
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