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
    double ticks;
    int MAX_MISSING_COUNT;

    VectorXd x_in = VectorXd(6); // state vector
    VectorXd z_in = VectorXd(4); // measurement vector
    MatrixXd P_in = MatrixXd(6, 6); // state covariance matrix
    P_in << 1, 0, 0,    0,    0, 0,
            0, 1, 0,    0,    0, 0,
            0, 0, 1000, 0,    0, 0,
            0, 0, 0,    1000, 0, 0,
            0, 0, 0,    0,    1, 0,
            0, 0, 0,    0,    0, 1;

    MatrixXd F_in = MatrixXd(6, 6); // state transition matrix
    F_in << 1, 0, 1, 0, 0, 0,
            0, 1, 0, 1, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    MatrixXd H_in = MatrixXd(4, 6); // measurement matrix
    H_in << 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;
    
    MatrixXd R_in = MatrixXd(4, 4); // measurement covariance matrix
    R_in << 0.01, 0,
            0,    0.01;

    MatrixXd Q_in = MatrixXd(6, 6); // process covariance matrix
    Q_in << 1, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0,
            0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 1;

    void cameraCallback(const sensor_msgs::ImageConstPtr &img);
    void depthCallback(const sensor_msgs::ImageConstPtr &depth_img);
    void track(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people);
    double centroidDistance(cv::Point a, cv::Point b);
    double iouScore(cv::Rect pred_box, cv::Rect detect_box);
    VectorXd getMeasurementVector(const darknet_ros_msgs::BoundingBox& box);
};

tracker::tracker() {

  ros::NodeHandle handler;
  ros::NodeHandle private_handler;

  string camera_topic;
  string depth_camera_topic;
  string darknet_topic;
  string person_topic;
  
  private_handler.getParam("camera_topic", camera_topic);
  private_handler.getParam("depth_camera_topic", depth_camera_topic);
  private_handler.getParam("darknet_topic", darknet_topic);
  private_handler.getParam("person_topic", person_topic);
  private_handler.getParam("MAX_MISSING_COUNT", MAX_MISSING_COUNT);

  camera_sub = handler.subscribe(camera_topic, 1, &tracker::cameraCallback, this);
  darknet_sub = handler.subscribe(darknet_topic, 1, &tracker::track, this);
  // depth calculation may differ from camera to camera, i.e the formula 
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

VectorXd getMeasurementVector(const darknet_ros_msgs::BoundingBox& box) {

  VectorXd z = VectorXd(4);
  z << box.xmax + box.xmin / 2,
       box.ymax + box.ymin / 2,
       box.width,
       box.height;

  return z;

}

void cameraCallback(const sensor_msgs::ImageConstPtr &img) {
  cv_ptr = cv_bridge::toCvCopy(img, enc::BGR8);

  image_color = cv_ptr->image; //current frame
  has_image = true;
}

void depthCallback(const sensor_msgs::ImageConstPtr &depth_img) {
  for (int i = 0; i < ekf_list.size(); i++) {
    target_person.x = ekf_list.x_in(0);
    target_person.y = ekf_list.x_in(1);
    target_person.image_width = image_color.cols;
    target_person.x_vel = ekf_list.x_in(2);
    target_person.y_vel = ekf_list.x_in(3);

    depths = (float*)(&depth_img->data[0]);
    int center_idx = target_person.x + (image_color.cols * target_person.y); // Zed Depth 
    target_person.depth = depths[center_idx];

    pub.publish(target_person);
  }
}

void track(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {

  vector<darknet_ros_msgs::BoundingBoxes> box_list;
  vector< vector<double> > distance_matrix(ekf_list.size());
  vector< vector<double> > iou_matrix(ekf_list.size());
  vector<int> assignment_list;

  double prev_ticks = ticks;
  ticks = (double) cv::getTickCount();

  double dT = (ticks - prev_ticks) / cv::getTickFrequency();

  if (initial_ && has_image) {
    for (const darknet_ros_msgs::BoundingBoxes &bounding_box : people->bounding_boxes) {
      box_list.push_back(bounding_box);
      x_in << (bounding_box.xmax + bounding_box.xmin) / 2,
              (bounding_box.ymax + bounding_box.ymin) / 2,
              0,
              0,
              (bounding_box.width),
              (bounding_box.height);
      ExtKalmanFilter ekf;
      ekf.init(x_in, P_in, F_in, Q_in, H_in, R_in);
      ekf_list.push_back(ekf);
    }
    initial_ = false;
  } else if (!initial_ && has_image) {

    for (const darknet_ros_msgs::BoundingBoxes &bounding_box : people->bounding_boxes) {
      box_list.push_back(bounding_box);
    }
    
    // Prediction Step
    for (int i = 0; i < ekf_list.size(); i++) {
      
      ekf_list[i].F_(0,2) = dT;
      ekf_list[i].F_(1,3) = dT;
      
      // Update process covariance matrix
      ekf_list[i].Q_ = // bla bla

      ekf_list[i].predict(dT) 
    }

    for (int i = 0; i < ekf_list.size(); i++) {
      distance_matrix[i] = vector<double>(box_list.size());
      iou_matrix[i] = vector<double>(box_list.size());

      for (int j = 0; j < box_list.size(); j++) {
        // Distance Matrix calculations
        cv::Point box_centroid = cv::Point( (box_list[j].xmax + box_list[j].xmin)/2 , (box_list[j].ymax + box_list[j].ymin)/2 );
        cv::Point ekf_centroid = ekf_list[i].getCentroid(); // get predicted position
        distance_matrix[i][j] = centroid_distance(box_centroid, ekf_centroid);

        // Iou Matrix calculations
        cv::Rect pred_box = ekf_list[i].getBox();
        cv::Rect detect_box = cv::Rect( cv::Point(r_box.xmin, r_box.ymin), cv::Point(r_box.xmax, r_box.ymax) );
        iou_matrix[i][j] = iouScore(pred_box, detect_box);
      }
    }
    
    //Hungarian Algorithm for data association
    double cost = hungarian_algorithm.Solve(distance_matrix, assignment_list);

    for (int i = 0; i < ekf_list.size(); i++) {
      // avoid all non-assigned objects
      if (assignment_list[i] >= 0) {
        // check that minimum IoU criterion is fulfilled
        if (iou_matrix[i][assignment_list[i]] > 0.5) {
          // assign the measurement vector & update the states
          z_in = getMeasurementVector(box_list[assignment_list[i]]);
          ekf_list[i].update(z_in);

          if (!box_list.empty()) {
            // erase the new assigned detection from the list
            box_list.erase(box_list.begin() + assignment_list[i]);
          }

        } else { ekf_list[i].noDetection(); }
      } else { ekf_list[i].noDetection(); }
    }

    for (int i = 0; i < ekf_list.size(); i++) {
      // missing count to stop tracking the object
      // this value can be expoerimented with 
      if (ekf_list[i].missing_count > MAX_MISSING_COUNT)
        ekf_list.erase(ekf_list.begin() + i);
    }


    // add new object if it is untracked
    for (int i = 0; i < box_list.size(); i++) {
      ExtKalmanFilter ekf;
      x_in << (bounding_box.xmax + bounding_box.xmin) / 2,
              (bounding_box.ymax + bounding_box.ymin) / 2,
              0,
              0,
              (bounding_box.width),
              (bounding_box.height);

      ekf.init(x_in, P_in, F_in, Q_in, H_in, R_in);
      ekf_list.push_back(ekf);
    }
  
  }

  /* TO-DO 
   * display detections and tracks with the help of opencv
   */

}

int main(int argc, char** argv) {
  ros::init(argc, argv, "Tracking");
  tracker people_tracker;

  ros::spin();
}
