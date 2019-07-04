#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>

#include "ros/ros.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "vision_msgs/Detection2DArray.h"
#include "vision_msgs/Detection2D.h"
#include "vision_msgs/BoundingBox2D.h"
#include "geometry_msgs/Pose2D.h"

using namespace std;

namespace enc = sensor_msgs::image_encodings;

class detectnet_publisher {
  public:
    detectnet_publisher();
    ~detectnet_publisher();
  private:

    ros::Subscriber detections_sub;
    image_transport::Subscriber camera_sub;

    vector<vision_msgs::Detection2DArray> detections;

    cv_bridge::CvImagePtr cv_ptr;
    bool has_image = false;

    void cameraCallback(const sensor_msgs::ImageConstPtr &img);
    void detectionCallback(const vision_msgs::Detection2DArray::ConstPtr &detections);
    void publish();

};

detectnet_publisher::detectnet_publisher() {

  ros::NodeHandle handler;
  ros::NodeHandle private_handler("~");
  image_transport::ImageTransport it(handler);
  string camera_topic;
  ROS_ASSERT(private_handler.getParam("camera_topic", camera_topic));
  string detection_topic;
  ROS_ASSERT(private_handler.getParam("detection_topic", detection_topic));

  cv::namedWindow("detectnet");

  camera_sub = it.subscribe("/zed_node/rgb/image_rect_color", 1, &detectnet_publisher::cameraCallback, this);
  detections_sub = handler.subscribe("detectnet/detections", 1, &detectnet_publisher::detectionCallback, this);
}

detectnet_publisher::~detectnet_publisher() {
  cv::destroyWindow("detectnet");
}

void detectnet_publisher::cameraCallback(const sensor_msgs::ImageConstPtr &img) {
  cv_ptr = cv_bridge::toCvCopy(img, enc::BGR8);
  has_image = true;
}

void detectnet_publisher::detectionCallback(const vision_msgs::Detection2DArray::ConstPtr &detections) {
  
  vector<vision_msgs::Detection2D> people;
  int i = 1; 
  if (has_image) {
    for (const vision_msgs::Detection2D &person : detections->detections) {
      const geometry_msgs::Pose2D person_center = person.bbox.center;
      ROS_INFO_STREAM("person " << i << ":");
      ROS_INFO_STREAM(person_center.x << ", " << person_center.y);
      
      //people.push_back(person);
      cv::rectangle(cv_ptr->image, cv::Rect(person_center.x, person_center.y, person.bbox.size_x, person.bbox.size_y), cv::Scalar(130,55,55), 4);
    }

    cv::imshow("detectnet", cv_ptr->image);
    cv::waitKey(5);
  }

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "detectnet_publisher");
  detectnet_publisher dpub;
  
  ros::spin();
}
