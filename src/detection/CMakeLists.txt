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
)

include_directories( include ${catkin_INCLUDE_DIRS} ) 

add_executable(tutorial src/images_tutorial1.cpp)
target_link_libraries(tutorial ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})

add_executable(depth src/depth_calc.cpp)
target_link_libraries(depth ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(depth detection_generate_messages_cpp)

add_executable(humanDetection src/human_detection.cpp)
target_link_libraries(humanDetection ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(humanDetection detection_generate_messages_cpp)

add_executable(yolo_kalman src/yolo_kalman.cpp)
target_link_libraries(yolo_kalman ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(yolo_kalman detection_generate_messages_cpp)

add_executable(yolo_final src/yolo_ekf.cpp)
target_link_libraries(yolo_final ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(yolo_final detection_generate_messages_cpp)

add_executable(yolo_TKal src/yolo_ekf2.cpp src/TKalmanFilter.cpp include/TKalmanFilter.h)
target_link_libraries(yolo_TKal ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(yolo_TKal detection_generate_messages_cpp)

add_executable(yolo_iou 
  src/yolo_ekf3.cpp 
  src/TKalmanFilter.cpp 
  src/tracks.cpp
  include/TKalmanFilter.h
  include/tracks.h
)
target_link_libraries(yolo_iou ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(yolo_iou detection_generate_messages_cpp)

add_executable(yolo src/yolo_human_detection.cpp)
target_link_libraries(yolo ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${OpenCV_LIBS})
add_dependencies(yolo detection_generate_messages_cpp)
