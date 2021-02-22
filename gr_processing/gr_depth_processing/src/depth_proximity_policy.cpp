#include "gr_depth_processing/depth_processing.h"
#include <pluginlib/class_list_macros.h>
#include <iostream>

PLUGINLIB_EXPORT_CLASS(gr_depth_processing::MyNodeletClass, nodelet::Nodelet)

namespace gr_depth_processing
{

  void MyNodeletClass::onInit(){

    //Local NodeHandle
    ros::NodeHandle local_nh("~");// = getMTPrivateNodeHandle();
    local_nh.param<std::string>("global_frame", global_frame_, "base_link");

    bool run_action_server;
    std::string modal;
    local_nh.param<bool>("run_action_server", run_action_server, false);
    local_nh.param<std::string>("modal", modal, "median");

    ROS_WARN_STREAM("Run action server " << run_action_server);
    ROS_WARN_STREAM("Global frame " << global_frame_);
    ROS_WARN_STREAM("Selected modality " << modal);

    //MEGA HACK
    std::string color_camera_info, depth_camera_info, color_camera_frame, depth_camera_frame, boundingboxes_topic;
    auto args = getRemappingArgs();

    color_camera_info  =  "/camera/color/camera_info";
    color_camera_frame  = "/camera/color/image_raw";
    depth_camera_info  = "/camera/depth/camera_info";
    depth_camera_frame = "/camera/depth/image_rect_raw";
    boundingboxes_topic = "/darknet_ros/bounding_boxes";

    if (args.size() > 0){
      loadFromRemappings<std::string>(args,"color_info",color_camera_info);
      loadFromRemappings<std::string>(args,"depth_info",depth_camera_info);
      loadFromRemappings<std::string>(args,"color_frame",color_camera_frame);
      loadFromRemappings<std::string>(args,"depth_frame",depth_camera_frame);
      loadFromRemappings<std::string>(args,"bounding_boxes",boundingboxes_topic);
    }

    tf2_listener_= new  tf2_ros::TransformListener(tf_buffer_);

    //Select filters
    filterImage = &cv_filter;
    if (modal == "full"){
      registerImage = &register_pointclouds;
    }
    //registerImage = &register_ransac_pointclouds;
    if (modal == "median"){
      registerImage = &register_median_pointclouds;
    }
    //registerImage = &cuda_register_median_pointclouds;

    ros::NodeHandle nh = getMTNodeHandle();
    max_range_ = 8.0;

    //Publisher to Proximity Monitor
    obstacle_pub_ = nh.advertise<geometry_msgs::PoseArray>("detected_objects",1);
    //Publish depth_image + distances
    depth_image_pub_ = nh.advertise<sensor_msgs::Image>("depth_image_processed", 1);
    //Publish FoundObjectArray using for training Models
    safety_pub_ = nh.advertise<safety_msgs::FoundObjectsArray>("found_object",1);


    if (!run_action_server){
      ROS_INFO_STREAM("Waiting for rgb and depth camera info: " << color_camera_info<<" " << depth_camera_info);
      camera_color_info_ = getOneMessage<sensor_msgs::CameraInfo>(color_camera_info);
      camera_depth_info_ = getOneMessage<sensor_msgs::CameraInfo>(depth_camera_info);
      ROS_INFO("Camera info received");
      color_image_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(nh, color_camera_frame, 2);
      depth_image_sub_ = new message_filters::Subscriber<sensor_msgs::Image>(nh, depth_camera_frame, 2);
      bounding_boxes_sub_ = new message_filters::Subscriber<darknet_ros_msgs::BoundingBoxes>(nh,boundingboxes_topic, 2);
      registered_syncronizer_ = new message_filters::Synchronizer<RegisteredSyncPolicy>(RegisteredSyncPolicy(2), *depth_image_sub_,*bounding_boxes_sub_);
      registered_syncronizer_->registerCallback(boost::bind(&MyNodeletClass::register_CB,this,_1,_2));
    }



    /*
    if(false){
      images_syncronizer_ = new message_filters::Synchronizer<ImagesSyncPolicy>(ImagesSyncPolicy(2), *color_image_sub_,*depth_image_sub_);
      images_syncronizer_->registerCallback(boost::bind(&MyNodeletClass::images_CB,this,_1,_2));
    }
    else{
      registered_syncronizer_ = new message_filters::Synchronizer<RegisteredSyncPolicy>(RegisteredSyncPolicy(2), *depth_image_sub_,*bounding_boxes_sub_);
      registered_syncronizer_->registerCallback(boost::bind(&MyNodeletClass::register_CB,this,_1,_2));
    }
    */
    else{
      aserver_ = boost::make_shared<actionlib::SimpleActionServer<gr_action_msgs::GRDepthProcessAction>>(nh, "gr_depth_process",
                                  boost::bind(&MyNodeletClass::execute_CB, this, _1), false);
                                  aserver_->start();
      ROS_INFO_STREAM("Run action server");
    }


    ROS_INFO("Depth Processing initialized");
  }


  bool MyNodeletClass::convertROSImage2Mat(cv::Mat& frame, const sensor_msgs::ImageConstPtr& ros_image){
    try{
      //cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::RGB8);
      //input_frame = cv_bridge::toCvCopy(depth_image, sensor_msgs::image_encodings::TYPE_16UC1)->image;
      frame = cv_bridge::toCvShare(ros_image, sensor_msgs::image_encodings::TYPE_16UC1)->image; //realsense
      //frame = cv_bridge::toCvShare(ros_image, sensor_msgs::image_encodings::TYPE_32FC1)->image; //zed
      return true;
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return false;
    }
  }

  void MyNodeletClass::publishOutput(cv::Mat frame, bool rotate){

    obstacle_pub_.publish(detected_objects_);
    safety_pub_.publish(objects_array_);

    sensor_msgs::Image out_msg;
    cv_bridge::CvImage img_bridge;
    std_msgs::Header header;

    try{
      //these lines are just for testing rotating image
      cv::Mat rot=cv::getRotationMatrix2D(cv::Point2f(0,0), 3.1416, 1.0);
      //cv::warpAffine(frame,frame, rot, frame.size());
      if (rotate){
        cv::rotate(frame,frame,1);
      }

      frame.convertTo(frame, CV_16UC1);

      //img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, frame);//COLOR
      //img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::TYPE_32FC1, frame);//zed
      img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::MONO16, frame);//realsense
      img_bridge.toImageMsg(out_msg); // from cv_bridge to sensor_msgs::Image
    }
    catch (cv_bridge::Exception& e){
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    depth_image_pub_.publish(out_msg);
  }


  void MyNodeletClass::images_CB(const sensor_msgs::ImageConstPtr color_image, const sensor_msgs::ImageConstPtr depth_image){
    boost::recursive_mutex::scoped_lock scoped_lock(mutex);
    cv::Mat process_frame;

    if (!convertROSImage2Mat(process_frame, depth_image)){//DEPTH
    //if (!convertROSImage2Mat(process_frame, color_image)){//COLOR
      return;
    }

    filterImage(process_frame);
    publishOutput(process_frame,false);

  }


  void MyNodeletClass::register_CB(const sensor_msgs::ImageConstPtr depth_image, const darknet_ros_msgs::BoundingBoxesConstPtr bounding_boxes){
    boost::recursive_mutex::scoped_lock scoped_lock(mutex);
    cv::Mat process_frame;
    std::vector<std::string> distance_to_objects;
    std::vector<std::pair<int,int>> objects_center;
    std::vector<cv::Rect> boundRect;

    const uint16_t * depth_array = reinterpret_cast<const uint16_t *>(&(depth_image->data[0]));


    if (!convertROSImage2Mat(process_frame, depth_image)){
      return;
    }

    cleanUpCycle();

    detected_objects_.poses.clear();
    double dist;

    //TODO THESE ARE CONSTANTS
    float center_x = camera_depth_info_.K[2];
    float center_y = camera_depth_info_.K[5];
    float constant_x = 1.0 /  camera_depth_info_.K[0];
    float constant_y = 1.0 /  camera_depth_info_.K[4];

    geometry_msgs::TransformStamped to_base_link_transform;

    objects_array_.objects.clear();

    for (auto it = bounding_boxes->bounding_boxes.begin(); it != bounding_boxes->bounding_boxes.end(); ++it){
      geometry_msgs::PoseStamped in, out;
      int center_row = it->xmin + (it->xmax - it->xmin)/2;
      int center_col = it->ymin + (it->ymax - it->ymin)/2;
      objects_center.push_back(std::make_pair(center_row, center_col));
      dist = registerImage(*it, process_frame, camera_depth_info_);

      in.header = depth_image->header;
      in.pose.orientation.w = 1.0;
      in.pose.position.x = (center_row - center_x) * dist * constant_x;
      in.pose.position.y = (center_col - center_y) * dist * constant_y;
      in.pose.position.z = dist;

      if (dist<0.1 || dist > max_range_){
        ROS_WARN_STREAM("Object out of range"<< dist);
        dist = gr_detection::OUT_OF_RANGE;
        //continue;
      }

      std::cout << "transform " << global_frame_ << " to " << in.header.frame_id << std::endl;
      to_base_link_transform = tf_buffer_.lookupTransform(global_frame_, in.header.frame_id, ros::Time(0), ros::Duration(1.0) );
      tf2::doTransform(in, out, to_base_link_transform);

      detected_objects_.header= out.header;
      detected_objects_.header.stamp = ros::Time::now();

      //THIS IF FOR GR_ML
      //Fill Object
      safety_msgs::Object object;
      //Copy header from transform useful to match timestamps
      objects_array_.header = out.header;
      //Copy centroid position
      object.pose = out.pose;

      //THIS IS FOR THE COMMON_DETECTION_UTILS
      gr_detection::Person person;
      //assuming global frame same of pointclouds
      person.pose = out.pose;

      std::cout << "Distance: " << dist <<  std::endl;

      std::cout << "before match" << std::endl;
      auto matchingid = matchDetection(person);
      object.object_id = matchingid;
      std::cout << "after match: " << matchingid << std::endl;

      if (!matchingid.empty() && matchingid.compare(gr_detection::NOPREVIOUSDETECTION)!=0){
        //GEt matched object
	      std::cout << "!" << std::endl;
        auto matched_object = GetObject(matchingid);
        auto nx = person.pose.position.x- matched_object.pose.position.x;
        auto ny = person.pose.position.y- matched_object.pose.position.y;
        auto nz = person.pose.position.z- matched_object.pose.position.z;

	      std::cout << "!!" << std::endl;
        tf2::Quaternion tf2_quat;
        //IF the distance is bigger than 5? cm then compute orientation and update
        if (std::abs(sqrt(nx*nx + ny*ny)) > 0.05 ){
          tf2_quat.setRPY(0,0, gr_detection::calculateYaw<double>(nx,ny,nz));
          person.pose.orientation = tf2::toMsg(tf2_quat);
        }
        else{
          //Reuse orientation
          person.pose.orientation = matched_object.pose.orientation;
        }

        out.pose.orientation = person.pose.orientation;
        //Updating
        ROS_INFO_STREAM("Updating person with id: " << matchingid);
        UpdateObject(matchingid, person);
      //copy class
      //objects_array_.objects.push_back(object);
      }
      else{
        ROS_WARN_STREAM("A new person has been found adding to the array");
        //testing map array_person (memory)
        insertNewObject(person);
      }
      //PUSH BACK FOR THE ACTION SERVER
      objects_array_.objects.push_back(object);
      //UPDATE ARRAY TO PROXIMITY RINGS
      detected_objects_.poses.push_back(out.pose);
      distance_to_objects.push_back(it->Class + std::to_string(dist));
      boundRect.push_back(cv::Rect(it->xmin, it->ymin, it->xmax - it->xmin, it->ymax - it->ymin));
    }

    ROS_INFO_STREAM("Detections read on camera");
    showCurrentDetections();

    //TO DRAW OUTPUT FRAME MAYBE WILL BE DELETED ON FUTURE VERSIONS
    auto it = distance_to_objects.begin();
    auto it2 = objects_center.begin();
    auto it3 = boundRect.begin();

    for (; it!= distance_to_objects.end(); ++it, ++it2, ++it3){
      cv::putText(process_frame, *it, cv::Point(it2->first, it2->second), cv::FONT_HERSHEY_PLAIN, 1,   0xffff , 2, 8);
      // cv::putText(process_frame, std::to_string(0.001*(depth_array[it2->first+ it2->second * process_frame.rows])), cv::Point(it2->first, it2->second+20), cv::FONT_HERSHEY_PLAIN,
      //                       1,   0xffff , 2, 8);
      cv::rectangle(process_frame, *it3, 0xffff);
    }
    publishOutput(process_frame, false);
  }


  void MyNodeletClass::execute_CB(const gr_action_msgs::GRDepthProcessGoalConstPtr &goal){
    camera_depth_info_ = goal->depth_info;
    gr_action_msgs::GRDepthProcessResult result;
    boost::shared_ptr<sensor_msgs::Image const> dimg = boost::make_shared<sensor_msgs::Image>(goal->depth_image);
    boost::shared_ptr<darknet_ros_msgs::BoundingBoxes const> bbs = boost::make_shared<darknet_ros_msgs::BoundingBoxes>(goal->bounding_boxes);
    register_CB(dimg, bbs);
    result.found_objects = objects_array_;
    aserver_->setSucceeded(result);
  }



}
