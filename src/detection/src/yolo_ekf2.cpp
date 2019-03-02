#include "ros/ros.h"
#include "iostream"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "detection/target_person.h"
#include "std_msgs/Int64.h"
#include "TKalmanFilter.hpp"

using namespace std;
using namespace cv;

namespace enc = sensor_msgs::image_encodings;

//int xMin, xMax, yMin, yMax;

darknet_ros_msgs::BoundingBox target_box;
detection::target_person target_person;

Mat image_color, image_hsv, depth_image;

bool has_target = false;
bool has_image = false;
int no_target_count = 0;

vector<const darknet_ros_msgs::BoundingBox> people_list;
vector<TKalmanFilter> kf_list;
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

string sensor_type;

ros::Publisher pub;

void getCvImage(const sensor_msgs::ImageConstPtr& img) {
    image_color = cv_bridge::toCvCopy(img, enc::BGR8)->image; //current frame
    cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
    has_image = true;
}

void getZedDepthImage(const sensor_msgs::ImageConstPtr& img) {
    depths = (float*)(&img->data[0]);

    target_person.x = center.x;
    target_person.x_vel = state.at<float>(2);
    target_person.y = center.y;
    target_person.y_vel = state.at<float>(3);
    target_person.image_width = image_color.cols;
    int center_idx = center.x + (image_color.cols * center.y); // Zed Depth 
    target_person.depth = depths[center_idx];

    pub.publish(target_person);
}


void getDepthImage(const sensor_msgs::ImageConstPtr& img) {
    depth_image = cv_bridge::toCvCopy(img, enc::TYPE_16UC1)->image; //current depth image

    target_person.x = center.x;
    target_person.x_vel = state.at<float>(2);
    target_person.y = center.y;
    target_person.y_vel = state.at<float>(3);
    target_person.image_width = image_color.cols;
    target_person.depth = depth_image.at<short int>(center);

    pub.publish(target_person);
}

Rect create_rect(const darknet_ros_msgs::BoundingBox& r_box) {
    return Rect(Point(r_box.xmin, r_box.ymin), Point(r_box.xmax, r_box.ymax));

}

Mat getHistogram(const darknet_ros_msgs::BoundingBox& box, const Mat& img_rgb) {

    int histSize[] = {30, 32};
    Mat roi_hsv, img_hsv, roi_hist;
    Rect r = Rect(Point(box.xmin, box.ymin), Point(box.xmax, box.ymax));
    Rect temp = r;
    if (r.area() > 95000) {
        r.height = r.height * 0.6;
        r.width = r.width * 0.6;
        r.x += (temp.width - r.width) / 2;
        r.y += (temp.height - r.height) / 2;
    }
    Mat roi_ = image_color(r);
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

void person_location(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {

    double best_score = 1.1;
    int best_box_i;
    int best_box_i_score;
    double maxDist = 100000000;
        
    Point center_t; //predicted point

    if (!has_image) {
        return;
    }

    if (has_target) {
        vector<const darknet_ros_msgs::BoundingBox> box_list;
        vector<const darknet_ros_msgs::BoundingBox> people_update_list; 
       
        for (int i = 0; i < people->bounding_boxes.size(); i++) {
            const darknet_ros_msgs::BoundingBox& box = people->bounding_boxes[i];
            if (box.probability > 0.5) 
                box_list.push_back(box);
        }

        vector< vector<double> > HungarianMat(kf_list.size());
        
        for (int i = 0; i < kf_list.size(); i++) {
            HungarianMat[i] = vector<double>(box_list.size());
            for (int j = 0; j < box_list.size(); j++) {
                HungarianMat[i][j] = calc_Distance( kf_list[i].getCenter(), getBboxCenter(box_list[j]) );
            }
        }
        
        people_list.clear();
        
        for (int i = 0; i < kf_list.size(); i++) {
            int minElementIndex = min_element(HungarianMat[i].begin(), HungarianMat[i].end()) - 
                                  HungarianMat[i].begin();
            people_list.push_back(HungarianMat[i][minElementIndex]);
        }
        
        /*
         * handling new people and people leaving
         */
        if (kf_list.size() > box_list.size()) {
            
        } else if (kf_list.size() < box_list.size()) {
            
        }

        for (int i = 0; i < kf_list.size(); i++) {
            kf_list[i].predict();
        }
        
        rectangle(image_color, create_rect(people_list[target_index]), Scalar(100,255,255), 4);
        imshow("Predict", image_color);
        waitKey(5);

        center_t = getBboxCenter(people_list[target_index]);
        target_person.x = center_t.x;
        target_person.y = center_t.y;
        target_person.image_width = image_color.cols;
        pub.publish(target_person);

        // ROS_INFO_STREAM("dist = " << maxDist);
        // ROS_INFO_STREAM("best_score = " << best_score);
        /*
         * Distance management, create a kalman class to allow for multiple person detection
         */
        // Rect target_rect = create_rect(people->bounding_boxes[best_box_i]);
        // double t_score = compareHist(target_roi, 
        //         getHistogram(people->bounding_boxes[best_box_i], image_color), CV_COMP_BHATTACHARYYA);

        // if (maxDist < 80 && t_score < 0.2) {
        //     target_box = people->bounding_boxes[best_box_i];
        //     target_roi = getHistogram(target_box, image_color); 
        // } else { 
        //     const darknet_ros_msgs::BoundingBox& t_box = people->bounding_boxes[best_box_i_score];
        //     Point t_box_center = Point( (t_box.xmax+t_box.xmin)/2 , (t_box.ymax+t_box.ymin)/2 ); 

        //     if ( calc_Distance(t_box_center, center_t) < 100 && best_score < 0.4 ) {
        //         cout << "Changed Target" << endl;
        //         target_box = people->bounding_boxes[best_box_i_score];
        //         target_roi = getHistogram(target_box, image_color); 
        //     } else {   
        //         target_box = people->bounding_boxes[best_box_i];
        //     }
        // }
    } else {

        for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {
            if (boxes.probability > 0.5) {
                people_list.push_back(boxes);
                if (target_box.probability < boxes.probability) {
                    target_box = boxes;
                }
            }
        }

        vector<int>::iterator it = find(people_list.begin(), people_list.end(), target_box);
        target_index = distance(people_list.begin(), it);

        for (int i = 0; i < people_list.size(); i++) {
            TKalmanFilter kf(6, 4);
            kf_list.push_back(kf);
        }
    }

    target_roi = getHistogram(target_box, image_color);
    cout << "INITIALIZED" << endl;

    if (people->bounding_boxes.empty()) {
        no_target_count++;
        cout << no_target_count << endl;
        if (no_target_count > 50) {
            has_target = false;
        }

    } else {
        no_target_count = 0;

        if (!has_target) { //first detection!     
            for (int i = 0; i < kf_list.size(); i++) {
                kf_list[i].init(people_list[i]);
            }
            has_target = true;
        } else {
            for (int i = 0; i < kf_list.size(); i++) {
                kf_list[i].update(people_list[i]);
            }
        }
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "people_listener");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~"); 
    image_transport::ImageTransport it(nh);
    namedWindow("Predict");

    pub = nh.advertise<detection::target_person>("/person", 1);
    nh_private.getParam("sensor_type", sensor_type);

    image_transport::Subscriber RGBImage_sub = it.subscribe("/image_color", 1, getCvImage);
    ros::Subscriber sub = nh.subscribe("/darknet_ros/bounding_boxes", 100, person_location);

    if (sensor_type.compare("kinect") == 0) {
        image_transport::Subscriber depthImage_sub = it.subscribe("/image_depth", 1, getDepthImage);
    } else {
        image_transport::Subscriber depthImage_sub = it.subscribe("/image_depth", 1, getZedDepthImage);
    }

    startWindowThread();
    ros::spin();

    return 0;
}
