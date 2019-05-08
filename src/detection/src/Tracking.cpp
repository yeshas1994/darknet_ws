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
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"
#include "detection/target_person.h"
#include "std_msgs/Int64.h"
#include "detection/Hungarian.h"
#include "detection/EKF.h"

using namespace std;
using namespace Eigen;

namespace enc = sensor_msgs::image_encodings;

class tracker {
  public:
    tracker();

  private:
    double max_invis_count;

    ros::Subscriber camera_sub;
    ros::Subscriber depth_sub;
    ros::Subscriber darknet_sub;

    ros::Publisher person_pub;

    darknet_ros_msgs::BoundingBox target_box;
    detection::target_person target_person;
    HungarianAlgorithm hungarian_algorithm;
    cv_bridge::CvImagePtr cv_ptr;
    vector<ExtKalmanFilter> ekf_list;
    float* depths;
    bool has_image= false;
    bool initial_ = true;
    cv::Mat image_color;

    void cameraCallback(const sensor_msgs::ImageConstPtr &img);
    void depthCallback(const sensor_msgs::ImageConstPtr &depth_img);
    void track(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people);
    double centroidDistance(cv::Point a, cv::Point b);
    double iouScore(cv::Rect pred_box, cv::Rect detect_box);
};

tracker::tracker() {

  ros::NodeHandle handler;
  ros::NodeHandle private_handler;

  string camera_topic;
  ROS_ASSERT(private_handler.getParam("camera_topic", camera_topic));
  string depth_camera_topic;
  ROS_ASSERT(private_handler.getParam("depth_camera_topic", depth_camera_topic));
  string darknet_topic;
  private_handler.param("darknet_topic", darknet_topic, "/darknet/bounding_boxes");
  string person_topic;
  ROS_ASSERT(private_handler.getParam("person_topic", person_topic));

  camera_sub = handler.subscribe(camera_topic, 1, &tracker::cameraCallback, this);
  darknet_sub = handler.subscribe(darknet_topic, 1, &tracker::track, this);
  depth_sub = handler.subscribe(depth_camera_topic, 1, &tracker::depthCallback, this);

  person_pub = handler.advertise<detection::target_person>(person_topic, 1, true);
}

double centroid_distance(cv::Point A, cv::Point b) {
  return sqrt( pow((A.x - B.x),2) + pow((A.y - B.y),2) );
}

double iouScore(cv::Rect pred_box, cv::Rect detect_box) {
  
  double xmin = max(predBox.x, detectBox.x);
  double ymin = max(predBox.y, detectBox.y);
  double xmax = min(predBox.x + predBox.width, detectBox.x + detectBox.width);
  double ymax = min(predBox.y + predBox.height, detectBox.y + detectBox.height);

  cv::Rect intersect(xmin, ymin, xmax-xmin, ymax-ymin);

  double iou = (double) intersect.area() / ( (double) predBox.area() + (double) detectBox.area() -
                                            (double) intersect.area() );

  return iou;
}

void cameraCallback(const sensor_msgs::ImageConstPtr &img) {
  cv_ptr = cv_bridge::toCvCopy(img, enc::BGR8);

  image_color = cv_ptr->image; //current frame
  has_image = true;
}

void depthCallback(const sensor_msgs::ImageConstPtr &depth_img) {
  target_person.x = ;
  target_person.y = ;
  target_person.image_width = ;
  target_person.x_vel = 
    target_person.y_vel = 

    depths = (float*)(&depth_img->data[0]);
  int center_idx = target_person.x + (image_color.cols * target_person.y); // Zed Depth 
  target_person.depth = depths[center_idx];

  pub.publish(target_person);
}

void track(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {

  vector<darknet_ros_msgs::BoundingBoxes> box_list;
  vector< vector<double> > distance_matrix(ekf_list.size());
  vector< vector<double> > iou_matrix(ekf_list.size());
  vectot<int> assignment_list;

  if (initial_) {
    for (const darknet_ros_msgs::BoundingBoxes &bounding_box : people->bounding_boxes) {
      box_list.push_back(bounding_box);
      ExtKalmanFilter ekf;
      ekf.init( /******/ );
      ekf_list.push_back(ekf);
    }
    initial_ = false;
  } else {

    for (const darknet_ros_msgs::BoundingBoxes &bounding_box : people->bounding_boxes) {
      box_list.push_back(bounding_box);
    }

    for (int i = 0; i < ekf_list.size(); i++) {
      ekf_list[i].predict(dT)
    }

    for (int i = 0; i < ekf_list.size(); i++) {
      distance_matrix[i] = vector<double>(box_list.size());
      iou_matrix[i] = vector<double>(box_list.size());

      for (int j = 0; j < box_list.size(); j++) {

        cv::Point box_centroid = cv::Point( (box_list[j].xmax + box_list[j].xmin)/2 ,                                                           (box_list[j].ymax + box_list[j].ymin)/2 );
        cv::Point ekf_centroid = ekf_list[i].get_centroid();
        distance_matrix[i][j] = centroid_distance(box_centroid, ekf_centroid);

        cv::Rect pred_box = ekf_list[i].getBox();
        cv::Rect detect_box = cv::Rect( cv::Point(r_box.xmin, r_box.ymin),                                                                  cv::Point(r_box.xmax, r_box.ymax) );
        iou_matrix[i][j] = iouScore(pred_box, detect_box);
      }
    }

    double cost = hungarian_algorithm.Solve(distance_matrix, assignment_list);

  }
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "Tracking");
  tracker people_tracker;

  ros::spin();
}