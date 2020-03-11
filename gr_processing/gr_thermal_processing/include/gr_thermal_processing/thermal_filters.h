#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video/background_segm.hpp>
#include <geometry_msgs/Accel.h>
#include <cv_bridge/cv_bridge.h>

#include <math.h>
#include <random>

#include <gr_thermal_processing/thermal_config.hpp>

void cv_filter(cv_bridge::CvImagePtr& frame, geometry_msgs::Accel& output, const gr_thermal_processing::ThermalFilterConfig* config){
    try{
        //rotate
        /*double angle = 180.0;
        cv::Point2f pt(frame->image.cols/2., frame->image.rows/2.);    
        cv::Mat r = cv::getRotationMatrix2D(pt, angle, 1.0);
        cv::warpAffine(frame->image, frame->image, r, frame->image.size());
        */
        //cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
        //im.at<uint16_t>(cell_x+cell_y*im.rows);
        //cv::cvtColor(frame->image, frame->image, cv::COLOR_BGR2GRAY );
        cv::Mat element; 
        //cv::erode(frame->image, frame->image, element);
        //cv::dilate(frame->image, frame->image, element);

        //Original image comin on RGB no idea why
       //   Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
       cv::Mat kernel;
       cv::Point anchor( config->anchor_point,config->anchor_point);
       //ddepth = -1;
      
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

       /*cv::Mat myKernel((cv::Mat_< uchar >(5, 5) <<
                                 0, 0,  .1,  0, 0,
                                  0,0, .1,  0, 0,
                                 .1,.1, 1 , 0, .1,
                                  0, .1, .1,  0, 0,
                                  0, 0,  .1,  0, 0));
        */
       for (int i =0; i<config->filter_iterations; i++){
           element = cv::getStructuringElement( cv::MORPH_RECT,
                                          cv::Size(2+i, 2+i ),
                                    anchor);
           cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
           cv::dilate(frame->image, frame->image, element, anchor, 1);
  

           kernel = cv::Mat::ones(config->kernel_size,config->kernel_size,CV_32F )/ (float)(pow(config->kernel_size,2));
            //cv::Laplacian(frame->image, frame->image, CV_16UC1, 1);
            //cv::Sobel(frame->image, frame->image, CV_16UC1,1, 1);
//            cv::filter2D(frame->image, frame->image,config->ddepth ,kernel,anchor,config->delta);//, BORDER_DEFAULT );
                        cv::filter2D(frame->image, frame->image,-1,kernel,anchor,config->delta);//, BORDER_DEFAULT );

       }

       element = cv::getStructuringElement( cv::MORPH_RECT,
                    cv::Size(2+config->erosion_factor, 2+config->erosion_factor),
                    anchor);
       cv::erode(frame->image, frame->image, element);
       //kernel_size = 3;
       //cv::Mat aaa;
       //cv::Canny(frame->image, aaa,0, 10000,3);
        // = Scalar::all(0);
        //aaa.copyTo(frame->image, aaa);
       // cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
      
       
       //cv::dilate(frame->image, frame->image, element);
       //cv::GaussianBlur(frame->image, frame->image, cv::Size(3,3), 1, 0, cv::BORDER_DEFAULT);
       //cv::erode(frame->image, frame->image, element);
       //
       
       //cv::bilateralFilter(frame->image, l1, 5, 50.0, 50.0);//, cv::BORDER_DEFAULT);
       //frame->image = l1;
       
       
       //threshold( src_gray, dst, threshold_value, max_BINARY_value,threshold_type );
       //cv::cvtColor(frame->image,frame->image,CV_GRAY2RGB);
       
       cv::Mat aux;
       
       if (config->apply_threshold){
           frame->image.convertTo(aux, CV_8UC1, 1/255.0, 0);
           cv::threshold(aux,aux, config->threshold, 255,0);
           aux.convertTo(frame->image, CV_16UC1, 255.0, 0);
       }
       //cv::adaptiveThreshold(aux,aux, 255.0,1,1,11,0.9);
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::filter2D(frame->image, frame->image, ddepth , kernel, anchor, delta);//, BORDER_DEFAULT );
       //cv::Laplacian(frame->image, frame->image, CV_8UC1, 3, 3,0);
        //cv::convertScaleAbs(frame->image, frame->image);
       cv::Scalar mu, sigma;
       cv::meanStdDev(frame->image, mu, sigma);

      // cv::Canny(frame->image, l1, 0,10, 7);
       //cv::Mat dst;
       //dst = cv::Scalar::all(0);
       //frame->image.copyTo(dstz, l1);
       //frame->image = dst;

       //three channels mean is practically same somehow
       //normalizing
       //mu[0] /= 255;
       //sigma[0]/=255;

       float norm_factor = config->norm_factor;

       output.linear.x = norm_factor/mu[0];
       output.angular.x = norm_factor*sigma[0];

       output.linear.y = log(mu[0]);
       output.angular.y = norm_factor / exp(sigma[0]);

       output.angular.z = exp(0.001/sigma[0]);//output.angular.z/log(sigma[2]) + 0.1;
       //std::cout << output.linear.x * output.linear.y * output.linear.z * output.angular.x * output.angular.y * output.angular.z << std::endl;
       //std::cout << output.linear.x * output.linear.y * output.angular.x * output.angular.y * output.angular.z << std::endl;
       //output.linear.z =  log(output.angular.y * output.linear.y);///log(mu[2]) + 0.1;
       output.linear.z = exp(0.1 / mu[0]) ;
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