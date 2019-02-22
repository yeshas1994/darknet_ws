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

//int xMin, xMax, yMin, yMax;

darknet_ros_msgs::BoundingBox target_box;
detection::target_person target_person;

Mat image_color, image_hsv, depth_image;

bool has_target = false;
bool has_image = false;
int no_target_count = 0;

// Kalman initialization
KalmanFilter kf(6, 4, 0, CV_32F); // initialize the kalman 
Mat state(6, 1, CV_32F);
Mat meas(4, 1, CV_32F);

double ticks = 0; // for dT

// cam shift stuff
Mat target_roi;
Rect track_window;
float sranges[] = {0 , 256};
float hranges[] = {0 , 180};
int channel[] = {0,1};
const float* ranges[] = {hranges, sranges};

ros::Publisher pub;

void getCvImage(const sensor_msgs::ImageConstPtr& img) {
    image_color = cv_bridge::toCvCopy(img, enc::BGR8)->image; //current frame
    cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
    has_image = true;
}

void getDepthImage(const sensor_msgs::ImageConstPtr& img) {
    depth_image = cv_bridge::toCvCopy(img, enc::TYPE_16UC1)->image; //current depth image
}

Mat getHistogram(const darknet_ros_msgs::BoundingBox& box, const Mat& img_rgb) {

    int histSize[] = {30, 32};
    Mat roi_hsv, img_hsv, roi_hist;
    Rect r = Rect(Point(target_box.xmin+10, target_box.ymin+10), 
                  Point(target_box.xmax-10, target_box.ymax-10));
    track_window = r;
    Mat roi_ = image_color(r);
    cvtColor(roi_, roi_hsv, COLOR_BGR2HSV);    
    calcHist(&roi_hsv, 1, channel, Mat(), roi_hist, 2, histSize, ranges, true, false);
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);

    return roi_hist;
}

void initializeKalmanFilter() {

    int state_size = 6;
    int meas_size = 4;
    int cont_size = 0;

    unsigned int type = CV_32F;


    //    Mat state(state_size, 1, type); // [x, y, v_x, v_y, w, h] state matrix
    //    Mat meas(meas_size, 1, type); // [z_x, z_y, z_w, z_h] these represent the matrix for the measured vals
    kf.statePost.at<float>(0) = 0;
    kf.statePost.at<float>(1) = 0;
    kf.statePost.at<float>(2) = 0;
    kf.statePost.at<float>(3) = 0;
    kf.statePost.at<float>(4) = 0;
    kf.statePost.at<float>(5) = 0;
    // Measure Matrix H
    // [1 0 0 0 0 0]
    // [0 1 0 0 0 0]
    // [0 0 0 0 1 0]
    // [0 0 0 0 0 1]

    kf.measurementMatrix = Mat::zeros(meas_size, state_size, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Transition State Matrix A
    // Note: set dT while processing
    // dT added in estimation loop 
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    setIdentity(kf.transitionMatrix);

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    setIdentity(kf.measurementNoiseCov, cv::Scalar(1e-1));

}

void person_location(const darknet_ros_msgs::BoundingBoxes::ConstPtr &people) {
    double best_score = -1;
    int best_box_i;

    double prev_ticks = ticks;
    ticks = (double) getTickCount();

    double dT = (ticks - prev_ticks) / getTickFrequency(); // converting tick count to seconds

    if (!has_image) {
        return;
    }

    if (has_target) {

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

        if (best_score > 0) {
            target_box = people->bounding_boxes[best_box_i];
        }

        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;

        //ROS_INFO_STREAM("dT = " << dT);
        
        double x_ =  kf.statePost.at<float>(0);
        double y_ = kf.statePost.at<float>(1);
        double dx_ = kf.statePost.at<float>(2);
        double dy_ = kf.statePost.at<float>(3);
        double w = kf.statePost.at<float>(4);
        double h = kf.statePost.at<float>(5);
    
        Mat prediction = kf.predict();

        kf.statePre.at<float>(0) = x_ + dx_ * dT;
        kf.statePre.at<float>(1) = y_ + dy_ * dT;
        kf.statePre.at<float>(2) = dx_;
        kf.statePre.at<float>(3) = dy_;
        kf.statePre.at<float>(4) = w; 
        kf.statePre.at<float>(5) = h;

        Rect predRect; //predicted state
        predRect.width = state.at<float>(4);
        predRect.height = state.at<float>(5);
        predRect.x = state.at<float>(0) - predRect.width / 2;
        predRect.y = state.at<float>(1) - predRect.height / 2;

        Point center; //predicted point
        center.x = state.at<float>(0);
        center.y = state.at<float>(1);

        target_person.x = center.x;
        target_person.y = center.y;
        target_person.image_width = image_color.cols;
      //  target_person.depth = depth_image.at<short int>(center);
        pub.publish(target_person);
    } else {
        for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {
            if (!has_target || boxes.probability > target_box.probability) {
                target_box = boxes;
            }

        }
        target_roi = getHistogram(target_box, image_color);
    }

    if (people->bounding_boxes.empty()) {
        no_target_count++;
        cout << no_target_count << endl;
        if (no_target_count > 50) {
            has_target = false;
        }

    } else {
        no_target_count = 0;

        meas.at<float>(0) = (target_box.xmin + target_box.xmax)/2.0;
        meas.at<float>(1) = (target_box.ymin + target_box.ymax)/2.0;
        meas.at<float>(2) = (float) (target_box.xmax - target_box.xmin);
        meas.at<float>(3) = (float) (target_box.ymax - target_box.ymin);

        if (!has_target) { //first detection!     

            //initialize the error and state
            kf.errorCovPre.at<float>(0) = 1; // px
            kf.errorCovPre.at<float>(7) = 1; // px
            kf.errorCovPre.at<float>(14) = 1;
            kf.errorCovPre.at<float>(21) = 1;
            kf.errorCovPre.at<float>(28) = 1; // px
            kf.errorCovPre.at<float>(35) = 1; // px

            state.at<float>(0) = meas.at<float>(0);
            state.at<float>(1) = meas.at<float>(1);
            state.at<float>(2) = 0;
            state.at<float>(3) = 0;
            state.at<float>(4) = meas.at<float>(2);
            state.at<float>(5) = meas.at<float>(3);

            kf.statePost = state;

            has_target = true;

        } else {

            Mat estimate = kf.correct(meas);
            kf.temp5.at<float>(0) = meas.at<float>(0) - kf.statePre.at<float>(0);
            kf.temp5.at<float>(1) = meas.at<float>(1) - kf.statePre.at<float>(1); 
            kf.temp5.at<float>(2) = meas.at<float>(2) - kf.statePre.at<float>(4); 
            kf.temp5.at<float>(3) = meas.at<float>(3) - kf.statePre.at<float>(5);
            kf.statePost = kf.statePre + kf.gain * kf.temp5;
            cout << "Estimate" << estimate << endl;
            cout << "statePost" << kf.statePost << endl;
            cout << "statePre" << kf.statePre << endl;
        }

    }

    //target_person.x = (target_box.xmin + target_box.xmax)/2;
    //target_person.y = (target_box.ymin + target_box.ymax)/2;
    //target_person.depth = depth_image.at<short int>(Point(target_person.x, target_person.y));
    //target_person.image_width = image_color.cols;
    //pub.publish(target_person);

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "people_listener");
    ros::NodeHandle nh;
    pub = nh.advertise<detection::target_person>("/person", 1);

    initializeKalmanFilter();

    image_transport::ImageTransport it(nh);
    namedWindow("lol");
    image_transport::Subscriber RGBImage_sub = it.subscribe("/usb_cam/image_raw", 1, getCvImage);
    //image_transport::Subscriber depthImage_sub = it.subscribe("/kinect2/qhd/image_depth_rect", 1, getDepthImage);
    ros::Subscriber sub = nh.subscribe("/darknet_ros/bounding_boxes", 1, person_location);
    startWindowThread();
    ros::spin();

    return 0;
}
