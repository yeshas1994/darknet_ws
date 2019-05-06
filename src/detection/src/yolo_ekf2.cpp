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
#include "detection/TKalmanFilter.h"
#include "detection/Hungarian.h"

using namespace std;
using namespace cv;

namespace enc = sensor_msgs::image_encodings;

//int xMin, xMax, yMin, yMax;

darknet_ros_msgs::BoundingBox target_box;
detection::target_person target_person;

Mat image_color, image_hsv, depth_image, image_fg;
Ptr<BackgroundSubtractor> MOG = createBackgroundSubtractorMOG2(500, 16, false);

bool has_target = false;
bool single_target = false;
bool has_image = false;
int no_target_count = 0;
bool chase = false;

vector<darknet_ros_msgs::BoundingBox> people_list;
vector<TKalmanFilter> kf_list;
TKalmanFilter kf_single;
int target_index;
// Kalman initialization
// KalmanFilter kf(6, 4, 0, CV_32F); // initialize the kalman 
// Mat state(6, 1, CV_32F);
// Mat meas(4, 1, CV_32F);
Point center;
double ticks = 0; // for dT

// cam shift stuff
Mat target_roi;
Rect track_window;
float sranges[] = {0 , 256};
float hranges[] = {0 , 180};
int channel[] = {0,1};
const float* ranges[] = {hranges, sranges};
float* depths;
double score;

// hungarian algo
double dist;
HungarianAlgorithm HungAlgo;

string sensor_type;
string method;
int counter = 0;

ros::Publisher pub;
cv_bridge::CvImagePtr cv_ptr;

void getCvImage(const sensor_msgs::ImageConstPtr& img) {
  cv_ptr = cv_bridge::toCvCopy(img, enc::BGR8);
  Mat image_temp;

  image_color = cv_ptr->image; //current frame
  cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
  MOG->apply(cv_ptr->image, image_temp);
  cv_ptr->image.copyTo(image_fg, image_temp);
  has_image = true;
}

void getZedDepthImage(const sensor_msgs::ImageConstPtr& img) {
  depths = (float*)(&img->data[0]);
  if (chase) {
    target_person.x = kf_single.kf_.statePost.at<float>(0);
    target_person.y = kf_single.kf_.statePost.at<float>(1);
    target_person.image_width = image_color.cols;
    target_person.x_vel = kf_single.kf_.statePost.at<float>(2);
    target_person.y_vel = kf_single.kf_.statePost.at<float>(3);
    int center_idx = target_person.x + (image_color.cols * target_person.y); // Zed Depth 
    target_person.depth = depths[center_idx];

    pub.publish(target_person);
  } else {

    pub.publish(target_person);
  }
}

void getDepthImage(const sensor_msgs::ImageConstPtr& img) {
  depth_image = cv_bridge::toCvCopy(img, enc::TYPE_16UC1)->image; //current depth image

  for (int i = 0; i < kf_list.size(); i++) {
    target_person.x = kf_list[target_index].kf_.statePre.at<float>(0);
    target_person.y = kf_list[target_index].kf_.statePre.at<float>(1);
    target_person.image_width = image_color.cols;
    target_person.depth = depth_image.at<short int>(Point(target_person.x, target_person.y));

    pub.publish(target_person);
  }
}

Rect create_rect(const darknet_ros_msgs::BoundingBox& r_box) {
  return Rect(Point(r_box.xmin, r_box.ymin), Point(r_box.xmax, r_box.ymax));

}

Mat getHistogram(const darknet_ros_msgs::BoundingBox& box, const Mat& img_rgb) {

  int histSize[] = {30, 32};
  Mat roi_hsv, img_hsv, roi_hist;
  Rect r = Rect(Point(box.xmin, box.ymin), Point(box.xmax, box.ymax));
  //Rect temp = r;
  //if (r.area() > 95000) {
  //  r.height = r.height * 0.6;
  //  r.width = r.width * 0.6;
  //  r.x += (temp.width - r.width) / 2;
  //  r.y += (temp.height - r.height) / 2;
  //}
  Mat roi_ = image_fg(r);

  cvtColor(roi_, roi_hsv, COLOR_BGR2HSV);    
  calcHist(&roi_hsv, 1, channel, Mat(), roi_hist, 2, histSize, ranges, true, false);
  normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

  return roi_hist;
}

double calc_Distance(Point a, Point b) {
  return sqrt( pow((a.x - b.x),2) + pow((a.y - b.y),2) );
}

Point getBboxCenter(const darknet_ros_msgs::BoundingBox& box) {
  return Point( (box.xmax+box.xmin)/2.0 , (box.ymax+box.ymin)/2.0 ); 
}

double iouScore(Rect predBox, Rect detectBox) {

  double xmin = max(predBox.x, detectBox.x);
  double ymin = max(predBox.y, detectBox.y);  
  double xmax = min(predBox.x + predBox.width, detectBox.x + detectBox.width);
  double ymax = min(predBox.y + predBox.height, detectBox.y + detectBox.height);

  Rect intersect(xmin, ymin, xmax-xmin, ymax-ymin);

  double iou = (double) intersect.area() / ((double) predBox.area() + (double) detectBox.area() - 
      (double) intersect.area());


  return iou;
}

void person_location(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {

  double best_score = 1.1;
  int best_box_i;
  int best_box_i_score;
  double maxDist = 100000000;

  Point center_t; //predicted point

  double prev_ticks = ticks;
  ticks = (double) getTickCount();

  double dT = (ticks - prev_ticks) / getTickFrequency();

  if (!has_image) {
    return;
  }

  if (has_target) {
    vector<darknet_ros_msgs::BoundingBox> box_list;

    if (people->bounding_boxes.size() < 1) {
      return;
    }

    for (int i = 0; i < people->bounding_boxes.size(); i++) {
      const darknet_ros_msgs::BoundingBox& box = people->bounding_boxes[i];
      box_list.push_back(box);
    }

    for (int i = 0; i < kf_list.size(); i++) {
      kf_list[i].predict(dT);
    }

    int method_num = 1;

    if (method.compare("distance")) {
      method_num == 1;
    } else if (method.compare("iou")) {
      method_num == 3;
    } else if (method.compare("shift")) {
      method_num == 2;
    }

    vector< vector<double> > HungarianMat(kf_list.size());
    vector< vector<double> > iouMat(kf_list.size());
    for (int i = 0; i < kf_list.size(); i++) {
      HungarianMat[i] = vector<double>(box_list.size());
      iouMat[i] = vector <double>(box_list.size());
      for (int j = 0; j < box_list.size(); j++) {
        iouMat[i][j] = iouScore(kf_list[i].getPredictedBox(), create_rect(box_list[j]));

        switch(method_num) {
          case 1: 
            HungarianMat[i][j] = calc_Distance( kf_list[i].getCenter(), getBboxCenter(box_list[j]) );
            break;
          case 2:  
            HungarianMat[i][j] = compareHist(kf_list[i].histogram, 
                getHistogram(box_list[j], image_color) , CV_COMP_BHATTACHARYYA);
            break;
          case 3:
            HungarianMat[i][j] = iouScore(kf_list[i].getPredictedBox(), create_rect(box_list[j]));
            break;
        }
      }
    }


    vector<int> assignment;
    double cost = HungAlgo.Solve(HungarianMat, assignment);

    int start_size = box_list.size();
    cout << "start size: " << box_list.size() << endl;
    for (int i = 0; i < kf_list.size(); i++) {
      int minElementIndex;

      if (method_num < 3) {
        minElementIndex = min_element(HungarianMat[i].begin(), HungarianMat[i].end()) - 
          HungarianMat[i].begin();
      } else {
        minElementIndex = max_element(HungarianMat[i].begin(), HungarianMat[i].end()) - 
          HungarianMat[i].begin();
      }
      if (assignment[i] >= 0) {
        if (iouMat[i][assignment[i]] > 0.5) {
          //kf_list[i].update(box_list[minElementIndex]);
          //kf_list[i].updateDetection(create_rect(box_list[minElementIndex]));
          //kf_list[i].updateHistogram(getHistogram(box_list[minElementIndex], image_color));

          kf_list[i].update(box_list[assignment[i]]);
          kf_list[i].updateDetection(create_rect(box_list[assignment[i]]));
          kf_list[i].updateHistogram(getHistogram(box_list[assignment[i]], image_color));


          for (int j = 0; j < kf_list.size(); j++) {
            HungarianMat[j][minElementIndex] = 99999999;
          }

          if (!box_list.empty()) {
            box_list.erase(box_list.begin() + assignment[i]);
          }

        }else{
          kf_list[i].noDetection();
        }
      } else {
        kf_list[i].noDetection();
      }
    }

    for (int i = 0; i < kf_list.size(); i++) {
      if (kf_list[i].invisible_count > 10) {
        kf_list.erase(kf_list.begin() + i);
      }
    }
    counter++; 
    cout << "end size: " << box_list.size() << endl;
    for (darknet_ros_msgs::BoundingBox boxes : box_list) {
      bool assigned = true;
      cout << "Hello" << endl;
      for (int i = 0; i < kf_list.size(); i++) {
        //if (compareHist(kf_list[i].histogram, getHistogram(boxes, image_color)
        //      , CV_COMP_BHATTACHARYYA) < 0.5) 
        //{
        if (!kf_list[i].updated()) {
          //      kf_list[i].update(boxes);
          //      kf_list[i].updateDetection(create_rect(boxes));
          //      kf_list[i].updateHistogram(getHistogram(boxes, image_color));
          assigned = false;
          break;
          //    }
      }
      //}
    }
    if (!assigned) {
      TKalmanFilter kf_(6, 4, kf_list.back().id + 1);
      kf_.setFixed(6, 4, CV_32F);
      kf_.init(boxes);
      kf_.updateDetection(create_rect(boxes));
      kf_.updateHistogram(getHistogram(boxes, image_color));
      kf_list.push_back(kf_);
    }
  }

  // for (int i = 0; i < kf_list.size(); i++) {
  //   if (!(kf_list[i].updated())) {
  //     kf_list[i].noDetection();
  //   }
  // }


  sort(kf_list.begin(), kf_list.end(), [](const TKalmanFilter& lhs, const TKalmanFilter& rhs) 
      {
      return lhs.invisible_count < rhs.invisible_count;
      });

  for (int i =  0; i < start_size; i++) {
    if (kf_list[i].invisible_count == 0)
      rectangle(cv_ptr->image, kf_list[i].bbox, Scalar(150,55,55), 4);
    else {

      rectangle(cv_ptr->image, kf_list[i].getPredictedBox(), Scalar(50,255,255), 4);
    }
  }
 
  Mat video_frame = cv_ptr->image;
  String name = "frame" + to_string(counter) + ".png";
  imwrite("/home/nvidia/Videos/" + name, video_frame);
  waitKey(5);

} else {

  int i = 0;
  double prob = 0.0;
  for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {
    people_list.push_back(boxes);
    TKalmanFilter kf(6, 4, i);
    kf.setFixed(6, 4, CV_32F); 
    kf.init(boxes);
    kf.updateHistogram(getHistogram(boxes, image_color));
    kf.updateDetection(create_rect(boxes));
    kf_list.push_back(kf);

    if (boxes.probability > prob ) {
      target_box = boxes;
      prob = boxes.probability;
      target_index = i;
    }
    i++;
  }

  has_target = true;
}
}

void singlePersonLocation(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {

  double best_score = 1.1;
  int best_box_i;
  int best_box_i_score;
  double maxDist = 100000000;
  chase = true;

  Point center_t; //predicted point

  double prev_ticks = ticks;
  ticks = (double) getTickCount();


  if (!has_image) {
    return;
  }

  if (has_target) {
    vector<darknet_ros_msgs::BoundingBox> box_list;

    if (people->bounding_boxes.size() < 1) {
      return;
    }

    for (int i = 0; i < people->bounding_boxes.size(); i++) {
      const darknet_ros_msgs::BoundingBox& box = people->bounding_boxes[i];
      box_list.push_back(box);
    }

    double dT = (ticks - prev_ticks) / getTickFrequency();
    kf_single.predict(dT);

    int method_num = 1;

    if (method.compare("distance")) {
      method_num == 1;
    } else if (method.compare("iou")) {
      method_num == 3;
    } else if (method.compare("shift")) {
      method_num == 2;
    }

    vector<double> HungarianMat(box_list.size());
    vector<double> HistMat(box_list.size());
    vector<double> iouMat(box_list.size());
    for (int j = 0; j < box_list.size(); j++) {
      iouMat[j] = iouScore(kf_single.getPredictedBox(), create_rect(box_list[j]));

      switch(method_num) {
        case 1: 
          HungarianMat[j] = calc_Distance( kf_single.getCenter(), getBboxCenter(box_list[j]) );
          HistMat[j] = compareHist(kf_single.histogram, 
              getHistogram(box_list[j], image_color) , CV_COMP_BHATTACHARYYA);
          break;
        case 2: 
          break;
        case 3:
          HungarianMat[j] = iouScore(kf_single.getPredictedBox(), create_rect(box_list[j]));
          break;
      }
    }


    int minElementIndex;
    int minHistIndex;
    double min_d = 9999;
    if (method_num < 3) {

      for (int i = 0; i < box_list.size(); i++) {
        if (HungarianMat[i] < min_d) {
          min_d = HungarianMat[i];
          minElementIndex = i;
        }
      }

      minHistIndex = min_element(HistMat.begin(), HistMat.end()) - 
        HistMat.begin();

    } else {
      minElementIndex = max_element(HungarianMat.begin(), HungarianMat.end()) - 
        HungarianMat.begin();
    }

    cout << iouMat[minElementIndex] << endl;
    if (iouMat[minElementIndex] >= 0.6) {
      kf_single.update(box_list[minElementIndex]);
      kf_single.updateDetection(create_rect(box_list[minElementIndex]));
      kf_single.updateHistogram(getHistogram(box_list[minElementIndex], image_color));

      if (!box_list.empty()) {
        box_list.erase(box_list.begin() + minElementIndex);
      }
    } else if (!(kf_single.updated())) {
      kf_single.noDetection();
    }



    if (kf_single.inv_single_count > 10) {
      TKalmanFilter kf(6, 4, kf_single.id+1);
      kf.setFixed(6, 4, CV_32F); 
      kf.init(box_list[minHistIndex]);
      kf.updateHistogram(getHistogram(box_list[minHistIndex], image_color));
      kf.updateDetection(create_rect(box_list[minHistIndex]));
      kf_single = kf;
    }

    ++counter;
    string name = "frame_" + to_string(counter) + ".png";

    //int index = 0;
    //for (darknet_ros_msgs::BoundingBox boxes : box_list) {
      //  bool assigned = false;
      //if (compareHist(kf_single.histogram, getHistogram(boxes, image_color)
        //    , CV_COMP_BHATTACHARYYA) < 0.6 && kf_single.inv_single_count > 5) {
        //    //        if (kf_single.inv_single_count > 6) {
        //    TKalmanFilter kf(6, 4, index);
        //    kf.setFixed(6, 4, CV_32F); 
        //    kf.init(boxes);
        //    kf.updateHistogram(getHistogram(boxes, image_color));
        //    kf.updateDetection(create_rect(boxes));
        //    kf_single = kf;
        //    assigned = true;
        //    break;
      //}
      //  index++;
      //}
      //
      rectangle(cv_ptr->image, kf_single.bbox, Scalar(150,55,55), 4);
      //rectangle(cv_ptr->image, kf_single.getPredictedBox(), Scalar(50,255,255), 2);
      //
      //    imshow("Predict", cv_ptr->image);
      //    waitKey(5);

      Mat video_frame = cv_ptr->image;

      //imwrite("/home/nvidia/Videos/" + name, video_frame);

      waitKey(5);
    } else {

      int i = 0;
      double prob = 0.0;
      for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {

        people_list.push_back(boxes);
        TKalmanFilter kf(6, 4, i);
        kf.setFixed(6, 4, CV_32F); 
        kf.init(boxes);
        kf.updateHistogram(getHistogram(boxes, image_color));
        kf.updateDetection(create_rect(boxes));
        kf_list.push_back(kf);

        if (boxes.probability > prob ) {
          target_box = boxes;
          prob = boxes.probability;
          target_index = i;
        }
        i++;
      }
      kf_single = kf_list[target_index];
      has_target = true;
    }

  }



  int main(int argc, char** argv) {
    ros::init(argc, argv, "people_listener");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~"); 
    image_transport::ImageTransport it(nh);
    chase = false;
    //namedWindow("Predict");

    pub = nh.advertise<detection::target_person>("/person", 1);

    image_transport::Subscriber RGBImage_sub = it.subscribe("/image_color", 1, getCvImage);
    //ros::Subscriber sub = nh.subscribe("/darknet_ros/bounding_boxes", 1, person_location);
    ros::Subscriber sub2 = nh.subscribe("/darknet_ros/bounding_boxes", 1, singlePersonLocation);

    //image_transport::Subscriber depthImage_sub = it.subscribe("/image_depth", 1, getDepthImage);
    image_transport::Subscriber depthImage_sub = it.subscribe("/image_depth", 1, getZedDepthImage);
    startWindowThread();
    ros::spin();


    return 0;
  }
