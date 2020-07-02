cmake_minimum_required(VERSION 2.8.3)
project(persons_stuff)

add_compile_options(-std=c++17)
add_definitions(-Wno-deprecated-declarations)
add_definitions(-Wno-dev)

find_package(catkin REQUIRED
        COMPONENTS
        roscpp
        pcl_conversions
        geometry_msgs
        safety_msgs
        jsk_recognition_msgs
        tf2
        tf2_ros
)

find_package(PCL REQUIRED)

find_package(Boost REQUIRED COMPONENTS
  filesystem
)

include_directories(include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    roscpp
    pcl_conversions
    geometry_msgs
    safety_msgs
    jsk_recognition_msgs
    tf2
    tf2_ros
  DEPENDS
    PCL
)

add_definitions(${PCL_DEFINITIONS})

add_library(${PROJECT_NAME}
            src/persons_pcd_reader.cpp
)

add_executable(${PROJECT_NAME}_node
            src/persons_pcd_reader_node.cpp
)


target_link_libraries(${PROJECT_NAME}
   ${catkin_LIBRARIES}
   ${PCL_LIBRARIES}
 )

target_link_libraries(${PROJECT_NAME}_node
   ${PROJECT_NAME}
 )