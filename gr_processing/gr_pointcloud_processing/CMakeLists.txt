cmake_minimum_required(VERSION 2.8.3)
project(gr_pointcloud_processing)

add_compile_options(-std=c++14)
add_definitions(-Wno-deprecated-declarations)
add_definitions(-Wno-dev)

find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        pcl_conversions
        geometry_msgs
        jsk_recognition_msgs
        tf2
        tf2_ros
        dynamic_reconfigure
        common_detection_utils
        nodelet
)

find_package(PCL REQUIRED)


find_package(Boost REQUIRED COMPONENTS
  filesystem
)

generate_dynamic_reconfigure_options(
  cfg/PointCloud.cfg
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  /usr/local/include/pcl_cuda_tools/
)

link_directories(/usr/local/include/pcl_cuda_tools/)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    pcl_conversions
    geometry_msgs
    jsk_recognition_msgs
    tf2
    tf2_ros
    dynamic_reconfigure
    common_detection_utils
    nodelet
  DEPENDS
    PCL
)

add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME}
            src/pointcloud_processing.cpp
)


target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
   pcl_cuda_tools
 )

install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY cfg
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/cfg)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
