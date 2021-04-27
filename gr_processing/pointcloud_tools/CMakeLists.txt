cmake_minimum_required(VERSION 2.8.3)
project(pointcloud_tools)

add_compile_options(-std=c++11)

#find_package(CUDA REQUIRED)

find_package(catkin REQUIRED
    COMPONENTS
    geometry_msgs
    sensor_msgs
    roscpp
    #pcl_ros
    pcl_conversions
    dynamic_reconfigure
)

#find_package(cmake_modules REQUIRED)
find_package(PCL REQUIRED)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)



catkin_package(
  INCLUDE_DIRS include/${PROJECT_NAME}
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
    geometry_msgs
    roscpp
    #pcl_ros
    pcl_conversions
    dynamic_reconfigure
)


add_definitions(${PCL_DEFINITIONS})

add_executable(pointcloud_tools#(${PROJECT_NAME}
            src/pointcloud_tools.cpp

)

#add_dependencies(${PROJECT_NAME} ${catkin_EXPORTED_TARGETS}
#                 ${PROJECT_NAME}_gencfg
#)

target_link_libraries( ${PROJECT_NAME}
 ${catkin_LIBRARIES} ${PCL_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY include
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  PATTERN ".svn" EXCLUDE)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
