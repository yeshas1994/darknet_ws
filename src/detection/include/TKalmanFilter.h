#include "opencv2/video/tracking.hpp"
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"

using namespace std;
using namespace cv;

class TKalmanFilter {
    public: 
        KalmanFilter kf_;

        Mat state; //n x 1
        Mat meas; //m x 1
        int n;
        int m;
        unsigned int type = CV_32F;

        TKalmanFilter(int n_, int m_); 
        void setFixed(int n_, int m_, unsigned int type_);
        void update(const darknet_ros_msgs::BoundingBox& box);
        void predict(double dT);
        void init(const darknet_ros_msgs::BoundingBox& box); //initial state vectors;
        Point getCenter();
};
