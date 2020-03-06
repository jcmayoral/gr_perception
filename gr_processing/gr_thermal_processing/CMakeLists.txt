cmake_minimum_required(VERSION 2.8.3)
project(gr_thermal_processing)
add_definitions(-std=c++11)

set(CMAKE_CXX_FLAGS "-std=c++11")

find_package(catkin REQUIRED
        COMPONENTS
        dynamic_reconfigure
        roscpp
        safety_msgs
        sensor_msgs
        geometry_msgs
        std_msgs
        cv_bridge
)

find_package(OpenCV 3 REQUIRED)

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
        safety_msgs
        sensor_msgs
        geometry_msgs
        std_msgs
        cv_bridge
)


file(GLOB SOURCES src/thermal*.cpp)
add_library(${PROJECT_NAME}  ${SOURCES})
add_executable(${PROJECT_NAME}_node src/node_thermal.cpp)
add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS} ${PROJECT_NAME})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
)


install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

