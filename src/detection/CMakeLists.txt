cmake_minimum_required(VERSION 2.8.3)
project(detection)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)
## Find catkin macros and libraries

## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  image_geometry
  roscpp
  sensor_msgs
  std_msgs
  message_generation
  vision_msgs
  geometry_msgs
  darknet_ros REQUIRED
)

add_message_files(
  FILES
  target_person.msg
)

generate_messages(
  DEPENDENCIES
  sensor_msgs   
  std_msgs
)

catkin_package(
  INCLUDE_DIRS
    include
  CATKIN_DEPENDS 
    cv_bridge 
    image_transport 
    roscpp 
    sensor_msgs 
    std_msgs
    vision_msgs
    geometry_msgs
  LIBRARIES detect_lib
)

include_directories( include ${catkin_INCLUDE_DIRS} ) 

install(DIRECTORY include/detection/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h")

add_library(detect_lib
  src/TKalmanFilter.cpp
  src/Hungarian.cpp)
target_link_libraries(detect_lib ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS}) 

add_executable(yolo_TKal 
  src/yolo_ekf2.cpp 
  src/TKalmanFilter.cpp 
  #include/TKalmanFilter.h
  #include/Hungarian.h
)
target_link_libraries(yolo_TKal ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS} detect_lib)
add_dependencies(yolo_TKal detection_generate_messages_cpp)

add_executable(detectnet_publisher src/detectnet_publisher)
target_link_libraries(detectnet_publisher ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS}) 

install(TARGETS detect_lib
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION} 
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
)

