cmake_minimum_required(VERSION 2.8.3)
project(gr_fused_detectors)

add_compile_options(-std=c++14)

find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        common_detection_utils
        gr_pointcloud_processing
        nodelet
)

find_library(my_library REQUIRED)
find_library(PCL 1.9 REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

catkin_package(
  CATKIN_DEPENDS
    roscpp
    common_detection_utils
    gr_pointcloud_processing
    nodelet
)

add_library(${PROJECT_NAME} src/fused_detectors.cpp)
add_executable(${PROJECT_NAME}_node src/ros_wrapper_node.cpp)

target_link_libraries(${PROJECT_NAME}_node
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${PROJECT_NAME}
)

target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   ${PROJECT_NAME}
)
