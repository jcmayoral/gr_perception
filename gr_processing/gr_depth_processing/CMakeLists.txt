cmake_minimum_required(VERSION 2.8.3)
project(gr_depth_processing)
add_definitions(-std=c++14)

find_package(catkin REQUIRED
        COMPONENTS
        dynamic_reconfigure
        roscpp
        pluginlib
        safety_msgs
        sensor_msgs
        geometry_msgs
        std_msgs
        cv_bridge
        message_filters
        darknet_ros_msgs
        tf2
        tf2_ros
        common_detection_utils
        gr_action_msgs
        actionlib
)

find_package(OpenCV 3.4.9 EXACT REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

catkin_package(
    INCLUDE_DIRS include
    LIBRARIES gr_safety_policies
    CATKIN_DEPENDS
        dynamic_reconfigure
        roscpp
        pluginlib
        safety_msgs
        sensor_msgs
        geometry_msgs
        std_msgs
        cv_bridge
        message_filters
        darknet_ros_msgs
        tf2
        tf2_ros
        common_detection_utils
        gr_action_msgs
        actionlib
)

file(GLOB SOURCES src/*.cpp)
add_library(${PROJECT_NAME}  ${SOURCES})


add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS})
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} pcl_cuda_tools)


### INSTALLS
install(TARGETS gr_depth_processing
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

#install(TARGETS gr_depth_processing_node
#  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)

#install(DIRECTORY config
#  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
#)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
