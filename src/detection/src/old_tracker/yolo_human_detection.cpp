#include "ros/ros.h"
#include "iostream"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
//#include "sensor_msgs/PointCloud2"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include "detection/target_person.h"
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"
#include "std_msgs/Int64.h"

using namespace std;
using namespace cv;

namespace enc = sensor_msgs::image_encodings;

int xMin, xMax, yMin, yMax;

darknet_ros_msgs::BoundingBox target_box;
detection::target_person target_person;

bool has_target = false;
bool has_image = false;

Mat image_color, image_hsv, target_roi, depth_image;
Rect track_window;
float sranges[] = {0 , 256};
float hranges[] = {0 , 180};
int channel[] = {0,1};
const float* ranges[] = {hranges, sranges};


ros::Publisher pub;

void getCvImage(const sensor_msgs::ImageConstPtr& img) {
    image_color = cv_bridge::toCvCopy(img, enc::BGR8)->image;
    cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
    has_image = true;
}

void getDepthImage(const sensor_msgs::ImageConstPtr& img) {
    depth_image = cv_bridge::toCvCopy(img, enc::TYPE_16UC1)->image;
}

Mat getHistogram(const darknet_ros_msgs::BoundingBox& box, const Mat& img_rgb) {
        
    int histSize[] = {30, 32};
    Mat roi_hsv, img_hsv, roi_hist;
    Rect r = Rect(Point(xMin+10,yMin+10), Point(xMax-10,yMax-10));
    track_window = r;
    Mat roi_ = image_color(r);
    cvtColor(roi_, roi_hsv, COLOR_BGR2HSV);    
    calcHist(&roi_hsv, 1, channel, Mat(), roi_hist, 2, histSize, ranges, true, false);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    return roi_hist;

}

void person_location(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {
    Mat dst;
    double best_score = -1;
    int best_box_i;

    if (people->bounding_boxes.empty()) {
        return;
    }

    if (!has_image) {
        return;
    }

    if (!has_target) {     

        for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {
            if (!has_target || boxes.probability > target_box.probability) {
                target_box = boxes;
                xMin = (int)boxes.xmin;
                xMax = (int)boxes.xmax;
                yMin = (int)boxes.ymin;
                yMax = (int)boxes.ymax;
                ROS_INFO("%d, %d, %d, %d", xMin, xMax, yMin, yMax);
                ROS_INFO("%d, %d", image_color.cols, image_color.rows);
            }

            has_target = true;

        }
        target_roi = getHistogram(target_box, image_color);
    } else {
        for (int i = 0; i < people->bounding_boxes.size(); i++) {
            const darknet_ros_msgs::BoundingBox& box = people->bounding_boxes[i];

            if (box.probability < 0.5) {
                continue;
            }

            Mat box_histogram = getHistogram(box, image_color);
            double score = compareHist(box_histogram, target_roi, CV_COMP_BHATTACHARYYA);

            if (score > best_score) {
                best_score = score;
                best_box_i = i;
            }
        }
    }
    if (best_score > 0) 
        target_box = people->bounding_boxes[best_box_i];
    
   
    target_person.x = (target_box.xmin + target_box.xmax)/2;
    target_person.y = (target_box.ymin + target_box.ymax)/2;
    target_person.depth = depth_image.at<short int>(Point(target_person.x, target_person.y));
    target_person.image_width = image_color.cols;
    pub.publish(target_person);

    Point bl = Point(target_box.xmin,target_box.ymin);
    Point tr = Point(target_box.xmax,target_box.ymax);
    rectangle(image_color, Rect(bl, tr), CV_RGB(255,255,255), 1);

    
    /* 
       calcBackProject(&image_hsv, 1, channel, target_roi, dst, ranges);
       CamShift(dst, track_window, TermCriteria(TermCriteria::EPS | TermCriteria::COUNT, 10, 1));
    // cout << track_window << endl;
    rectangle(image_color, track_window, Scalar(255, 128, 128), 2);
    cout << track_window << endl;
    */

//imshow("lol", image_color);
//waitKey(10);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "people_listener");
    ros::NodeHandle nh;
    pub = nh.advertise<detection::target_person>("/person", 1);

    image_transport::ImageTransport it(nh);
    namedWindow("lol");
    image_transport::Subscriber RGBImage_sub = it.subscribe("/kinect2/qhd/image_color", 1, getCvImage);
    image_transport::Subscriber depthImage_sub = it.subscribe("/kinect2/qhd/image_depth_rect", 1, getDepthImage);
    ros::Subscriber sub = nh.subscribe("/darknet_ros/bounding_boxes", 100, person_location);
    startWindowThread();
    ros::spin();

    return 0;
}
