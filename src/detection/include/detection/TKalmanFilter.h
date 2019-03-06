#include "opencv2/video/tracking.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "darknet_ros_msgs/BoundingBoxes.h"
#include "darknet_ros_msgs/BoundingBox.h"

using namespace std;
using namespace cv;

class TKalmanFilter {
    public: 
        KalmanFilter kf_;
        int id; 
        int age;
        int invisible_count;
        int visible_count;
        Mat histogram;
        Rect bbox;
        Mat state; //n x 1
        Mat meas; //m x 1
        int n;
        int m;
        unsigned int type;
        bool assigned;

        TKalmanFilter(int n_, int m_); 
        TKalmanFilter(int n_, int m_, int id_); 
        TKalmanFilter();
        void setFixed(int n_, int m_, unsigned int type_);
        void update(const darknet_ros_msgs::BoundingBox& box);
        void predict(double dT);
        void init(const darknet_ros_msgs::BoundingBox& box); //initial state vectors;
        void updateHistogram(Mat hist_);
        void updateDetection(Rect bbox_);
        void noDetection();
        bool updated();
        Rect getPredictedBox();
        Point getCenter();
};
