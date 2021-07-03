cmake_minimum_required(VERSION 2.8.3)
project(common_detection_utils)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

#TODO create config gile of "magic thresholds"
find_package(YAML-CPP 0.6 REQUIRED)


## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
  roscpp
  geometry_msgs
  tf2
  tf2_geometry_msgs
)

###########
## Build ##
###########


add_library(${PROJECT_NAME} src/common_detection_utils.cpp)

add_executable(${PROJECT_NAME}_node src/fusion_detection_utils_node.cpp)
#add_dependencies(${PROJECT_NAME}_node ${catkin_EXPORTED_TARGETS})

target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES})
target_link_libraries(${PROJECT_NAME}_node ${PROJECT_NAME})


install(DIRECTORY ${CMAKE_SOURCE_DIR}/include/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(TARGETS  ${PROJECT_NAME} ${PROJECT_NAME}_node
   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
