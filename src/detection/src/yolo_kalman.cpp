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

KalmanFilter kf(6, 4, 0, CV_32F); // initialize the kalman 
Mat state(6, 1, CV_32F);
Mat meas(4, 1, CV_32F);

double ticks = 0; // for dT

ros::Publisher pub;

void getCvImage(const sensor_msgs::ImageConstPtr& img) {
    image_color = cv_bridge::toCvCopy(img, enc::BGR8)->image; //current frame
    cvtColor(image_color, image_hsv, COLOR_BGR2HSV);
    has_image = true;
}

void getDepthImage(const sensor_msgs::ImageConstPtr& img) {
    depth_image = cv_bridge::toCvCopy(img, enc::TYPE_16UC1)->image; //current depth image
}

void initializeKalmanFilter() {
	
	int state_size = 6;
	int meas_size = 4;
	int cont_size = 0;

    unsigned int type = CV_32F;


//    Mat state(state_size, 1, type); // [x, y, v_x, v_y, w, h] state matrix
    
    // Measure Matrix H
    // [1 0 0 0 0 0]
    // [0 1 0 0 0 0]
    // [0 0 0 0 1 0]
    // [0 0 0 0 0 1]
//    Mat meas(meas_size, 1, type); // [z_x, z_y, z_w, z_h] these represent the matrix for the measured vals

    kf.measurementMatrix = Mat::zeros(meas_size, state_size, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Transition State Matrix A
    // Note: set dT while processing
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
        kf.transitionMatrix.at<float>(2) = dT;
        kf.transitionMatrix.at<float>(9) = dT;

        //ROS_INFO_STREAM("dT = " << dT);
            
        state = kf.predict();
     //    Rect predRect; //predicted state
     //   predRect.width = state.at<float>(4);
     //   predRect.height = state.at<float>(5);
     //   predRect.x = state.at<float>(0) - predRect.width / 2;
     //   predRect.y = state.at<float>(1) - predRect.height / 2;
        
        Point center; //predicted point
        center.x = state.at<float>(0);
        center.y = state.at<float>(1);
                
        target_person.x = center.x;
        target_person.y = center.y;
        target_person.image_width = image_color.cols;
        target_person.depth = depth_image.at<short int>(center);
        pub.publish(target_person);
    }

    for (const darknet_ros_msgs::BoundingBox& boxes : people->bounding_boxes) {
        if (!has_target || boxes.probability > target_box.probability) {
            target_box = boxes;
        }
        
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
            
            kf.correct(meas);
            cout << meas << endl;
        }
           
    }

//    target_person.x = (target_box.xmin + target_box.xmax)/2;
//    target_person.y = (target_box.ymin + target_box.ymax)/2;
//    target_person.depth = depth_image.at<short int>(Point(target_person.x, target_person.y));
//    target_person.image_width = image_color.cols;
//    pub.publish(target_person);

}
int main(int argc, char **argv) {
    ros::init(argc, argv, "people_listener");
    ros::NodeHandle nh;
    pub = nh.advertise<detection::target_person>("/person", 1);

    initializeKalmanFilter();

    image_transport::ImageTransport it(nh);
//    namedWindow("lol");
    image_transport::Subscriber RGBImage_sub = it.subscribe("/kinect2/qhd/image_color", 1, getCvImage);
    image_transport::Subscriber depthImage_sub = it.subscribe("/kinect2/qhd/image_depth_rect", 1, getDepthImage);
    ros::Subscriber sub = nh.subscribe("/darknet_ros/bounding_boxes", 1, person_location);
    startWindowThread();
    ros::spin();

    return 0;
}
