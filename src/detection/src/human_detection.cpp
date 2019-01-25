/*#include "ros/ros.h"
#include "std_msgs/Int64.h"
#include "iostream"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
#include "sensor_msgs/image_encodings.h"
#include "opencv2/core/core.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "learning_depth/send.h"
#include "vector"
//#include <pcl_ros/point_cloud.h>
//#include <pcl/point_types>
//#include "sensor_msgs/PointCloud2"

namespace enc = sensor_msgs::image_encodings;

using namespace cv;
using namespace std;

cv_bridge::CvImagePtr cv_image;

Mat image;

void imageCallback(const sensor_msgs::ImageConstPtr& msg2) {
    cv_bridge::CvImagePtr cv_ptr;

    try {
        cv_ptr = cv_bridge::toCvCopy(msg2, enc::BGR8); //TYPE_16UC1
        cv_image = cv_ptr;
    }

    catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    //imshow("Tracking", cv_ptr->image);
}

class KinectSubscriber {
    ros::NodeHandle nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber sub = it.subscribe("/kinect2/qhd/image_color", 1, &KinectSubscriber::callback, &KSub);

    public:
        vector<Rect> bounding_box;
        //bool detected;
//        Mat img;

        void callback(const sensor_msgs::ImageConstPtr& msg2) {
            cv_bridge::CvImagePtr cv_ptr;

            try {
                cv_ptr = cv_bridge::toCvCopy(msg2, enc::BGR8); //TYPE_16UC1
                cv_image = cv_ptr;
            }

            catch (cv_bridge::Exception& e) {
                ROS_ERROR("cv_bridge exception: %s", e.what());
                return;
            }

            //HOGDescriptor hog(Size( 48, 96 ), Size( 16, 16 ), Size( 8, 8 ), Size( 8, 8 ), 9, 1, -1, HOGDescriptor::L2Hys, 0.2, false, cv::HOGDescriptor::DEFAULT_NLEVELS);
            //hog.winSize = Size(48, 96);
            //hog.setSVMDetector(HOGDescriptor::getDaimlerPeopleDetector());

            cout << "connected" << endl;

            HOGDescriptor hog;
            hog.setSVMDetector(HOGDescriptor::getDefaultPeopleDetector());

            vector<Rect> found, found_filtered;

            Mat img = cv_ptr->image;

            hog.detectMultiScale(img, found, 0, Size(8,8), Size(32,32), 2, false);

            for (size_t i = 0; i < found.size(); i++ ) {
                Rect r = found[i];
                size_t j;
                // Do not add small detections inside a bigger detection.
                for (j = 0; j < found.size(); j++) {
                    if (j != i && (r & found[j]) == r) {
                        break;
                    }
                }

                if (j == found.size()) {
                    found_filtered.push_back(r);
                }
            }

            for (size_t i = 0; i < found_filtered.size(); i++) {
                Rect r = found_filtered[i];
                // The HOG detector returns slightly larger rectangles than the real objects,
                // so we slightly shrink the rectangles to get a nicer output.
                r.x += cvRound(r.width*0.1);
                r.width = cvRound(r.width*0.8);
                r.y += cvRound(r.height*0.07);
                r.height = cvRound(r.height*0.8);
                //found_filtered.push_back(r);
                if (r.area() > 10000) {
                    cout << "detected" << endl;
                    rectangle(img, r, cv::Scalar(0,255,0), 3);
                    bounding_box.push_back(r);
                    Point rectCenter = (r.tl() + r.br()) * 0.5;
                    //circle(img, rectCenter, 1, cv::Scalar(0,0,255));
                }
            }
        }
};
*/
/*
   Mat frame;
   int stateSize = 6;
   int measSize = 4;
   int contrSize = 0;
   unsigned int type = CV_32F;

   KalmanFilter kf(stateSize, measSize, contrSize, type);
//Transition Matrix, Measurement Matrix
Mat state(stateSize, 1, type); //type = CV_32f
Mat meas(measSize, 1, type); //type = CV_32f
//Covariance Matrix
setIdentity(kf.transitionMatrix);

// Measure Matrix H
// 1 0 0 0 0 0
// 0 1 0 0 0 0
// 0 0 0 0 1 0
// 0 0 0 0 0 1
kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
kf.measurementMatrix.at<float>(0) = 1.0f;
kf.measurementMatrix.at<float>(7) = 1.0f;
kf.measurementMatrix.at<float>(16) = 1.0f;
kf.measurementMatrix.at<float>(23) = 1.0f;

// Process Noise Covariance Matrix Q
// Ex   0   0     0     0    0
// 0    Ey  0     0     0    0
// 0    0   Ev_x  0     0    0
// 0    0   0     Ev_y  0    0
// 0    0   0     0     Ew   0
kf.processNoiseCov.at<float>(0) = 1e-2;
kf.processNoiseCov.at<float>(7) = 1e-2;
kf.processNoiseCov.at<float>(14) = 5.0f;
kf.processNoiseCov.at<float>(21) = 5.0f;
kf.processNoiseCov.at<float>(28) = 1e-2;
kf.processNoiseCov.at<float>(35) = 1e-2;

//Measuring the noise covaraince matrix
setIdentity(kf.measurementNoiseCov, Scalar(1e-1));
//prediction rectangle and point can be determined from the state.
//Noise smoothing - blur/gaussian blur

//    stringstream sstr;
//    sstr << center.x << ", " <<  center.y;
//    putText(img, sstr, location, ...)



//kalman update for when found.size() == 0

//   int notFoundCount = 0;
bool detect = false;
double ticks = 0;

char ch = 0;


double precTick = ticks;
ticks = (double) getTickCount();

double dT = (ticks - precTick) / getTickFrequency();

Mat res = cv_ptr->image;

if (detect) {
// >>>> Matrix A
kf.transitionMatrix.at<float>(2) = dT;
kf.transitionMatrix.at<float>(9) = dT;
// <<<< Matrix A
cout << "dT:" << endl << dT << endl;

state = kf.predict();
cout << "State post:" << endl << state << endl;

Rect predRect;
predRect.width = state.at<float>(4);
predRect.height = state.at<float>(5);
predRect.x = state.at<float>(0) - predRect.width / 2;
predRect.y = state.at<float>(1) - predRect.height / 2;

Point predCenter;
predCenter.x = state.at<float>(0);
predCenter.y = state.at<float>(1);
circle(res, predCenter, 2, CV_RGB(255, 0, 0), -1);

rectangle(img, predRect, cv::Scalar(0,255,255), 3);
}

imshow("detection", img);

if (found_filtered.size() == 0) {
    notFoundCount++;
    cout << "notFoundCount" << notFoundCount << endl;
    if (notFoundCount > 100) {
        detect = false;
    }

} else {
    notFoundCount = 0;

    meas.at<float>(0) = found_filtered[0].x + found_filtered[0].width / 2; //center x
    meas.at<float>(1) = found_filtered[0].y + found_filtered[0].height / 2; //center y
    meas.at<float>(2) = (float)found_filtered[0].width; //box width
    meas.at<float>(3) = (float)found_filtered[0].height; //box height

    //First detection!
    if (!detect) {

        // >>>> Initialization
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
        // <<<< Initialization

        kf.statePost = state;

        detect = true;
    }
    else {
        kf.correct(meas); // Kalman Correction
    }

    //     cout << "Measure matrix: " << endl << meas << endl;

    // <<<< Kalman update

    //Final result
    ch = waitKey(1);
}


int main(int argc, char **argv) {
    ros::init(argc, argv, "human_detection");
    ros::NodeHandle nh;
    cout << "start" << endl;
    Mat frame;
    int stateSize = 6;
    int measSize = 4;
    int contrSize = 0;
    unsigned int type = CV_32F;
    //bool detected = false;

    KalmanFilter kf(stateSize, measSize, contrSize, type);
    //Transition Matrix, Measurement Matrix
    Mat state(stateSize, 1, type); //type = CV_32f
    Mat meas(measSize, 1, type); //type = CV_32f
    //Covariance Matrix
    setIdentity(kf.transitionMatrix);

    // Measure Matrix H
    // 1 0 0 0 0 0
    // 0 1 0 0 0 0
    // 0 0 0 0 1 0
    // 0 0 0 0 0 1
    kf.measurementMatrix = cv::Mat::zeros(measSize, stateSize, type);
    kf.measurementMatrix.at<float>(0) = 1.0f;
    kf.measurementMatrix.at<float>(7) = 1.0f;
    kf.measurementMatrix.at<float>(16) = 1.0f;
    kf.measurementMatrix.at<float>(23) = 1.0f;

    // Process Noise Covariance Matrix Q
    // Ex   0   0     0     0    0
    // 0    Ey  0     0     0    0
    // 0    0   Ev_x  0     0    0
    // 0    0   0     Ev_y  0    0
    // 0    0   0     0     Ew   0
    // 0    0   0     0     0    Eh
    kf.processNoiseCov.at<float>(0) = 1e-2;
    kf.processNoiseCov.at<float>(7) = 1e-2;
    kf.processNoiseCov.at<float>(14) = 5.0f;
    kf.processNoiseCov.at<float>(21) = 5.0f;
    kf.processNoiseCov.at<float>(28) = 1e-2;
    kf.processNoiseCov.at<float>(35) = 1e-2;
    //Measuring the noise covaraince matrix
    setIdentity(kf.measurementNoiseCov, Scalar(1e-1));
    //prediction rectangle and point can be determined from the state.
    //Noise smoothing - blur/gaussian blur

    //    stringstream sstr;
    //    sstr << center.x << ", " <<  center.y;
    //    putText(img, sstr, location, ...)
    //  kalman update for when found.size() == 0
    int notFoundCount = 0;
    bool detect = false;
    double ticks = 0;
    char ch = 0;

    //  namedWindow("lol");
    //  namedWindow("detection");
    //  imshow("lol", ::image);    
    //  startWindowThread();
    KinectSubscriber KSub;
    image_transport::ImageTransport it(nh);
    
    cout << "beginning image processing" << endl;
    while (ros::ok) {

        double precTicks = ticks;
        ticks = (double) getTickCount();

        double dT = (ticks - precTicks) / getTickFrequency();

        cout << "done" << endl;
        /*
        if (detect) {
            //Matrix A
            kf.transitionMatrix.at<float>(2) = dT;
            kf.transitionMatrix.at<float>(9) = dT;

            state = kf.predict();

            Rect prediction_Rect;
            prediction_Rect.width = state.at<float>(4); 
            prediction_Rect.height = state.at<float>(5);
            prediction_Rect.x = state.at<float>(0) - prediction_Rect.width / 2;
            prediction_Rect.y = state.at<float>(4) - prediction_Rect.height / 2;

            Point center;
            center.x = state.at<float>(0);
            center.y = state.at<float>(1);
            cout << center << endl;
        }

        if (KSub.bounding_box.size() == 0) {
            notFoundCount++;

            if (notFoundCount >= 100) {
                detect = false;
            }

        }

        else {
            notFoundCount = 0;

            meas.at<float>(0) = KSub.bounding_box[0].x + KSub.bounding_box[0].width / 2;
            meas.at<float>(1) = KSub.bounding_box[0].y + KSub.bounding_box[0].height / 2;
            meas.at<float>(2) = (float) KSub.bounding_box[0].width;
            meas.at<float>(3) = (float) KSub.bounding_box[0].height;

                    if (!detect) {
                    // >>>> Initialization 
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
                    // <<<< Initialization
                    kf.statePost = state;

                    detect = true;

                    }
                    else {
                        kf.correct(meas); 
                    }
                    ros::spin();
        }
        ros::spin();

    }
}

*/
