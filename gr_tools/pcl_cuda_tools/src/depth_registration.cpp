#include <iostream>
#include <math.h>
#include <pcl_cuda_tools/depth_registration.h>
#include <pcl_cuda_tools/cuda_functions.cuh>

DepthRegistration::DepthRegistration(cv::Mat image): frame_(image){
  std::cout << "Constructor cuda object"<<std::endl;
};

double DepthRegistration::run(){
  //std::lock_guard<std::mutex> lock(mtx);
  int  x[frame_.rows*frame_.cols];

  int width = frame_.cols;
  int height = frame_.rows;
  int _stride = frame_.step;//in case cols != strides

  float max_value = 65535;
  int bin_number = 1000;
  float delta = max_value/bin_number;

  for(int i = 0; i < height; i++){
      for(int j = 0; j < width; j++){
          x[ j * i + i] = static_cast<int>(frame_.at<ushort>(i,j))/delta;
          //printf("%d \n",x[j*i +i]);// << " , " <<  x[ j * i + i] <<  std::endl;
      }
  }

  /*

  for(int i = 0; i < height; i+=200){
      for(int j = 0; j < width; j+=200){
          std::cout << static_cast<unsigned>(frame_.at<unsigned char>(i,j)) <<std::endl;
      }
  }
  for (int r = 0; r < frame_rows; r++) {
    for (int c = 0; r < frame_rows; c++) {
      x[i] = static_cast<unsigned char>(M.at<unsigned char>(r,c));
  }
  */
  int n = frame_.rows * frame_.cols;

  int result = call_registration(x, n);
  return double(result*delta*0.001);

}

DepthRegistration::~DepthRegistration(){

}
