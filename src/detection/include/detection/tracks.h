#include <iostream>
#include "TKalmanFilter.h"
#include "darknet_ros_msgs/BoundingBox.h"
#include "opencv2/highgui/highgui.hpp" 
#include "opencv2/imgproc/imgproc.hpp" 
#include "opencv2/video/tracking.hpp" 
 
using namespace std;
using namespace cv;

class tracks {
  public:

    int id;
    Rect bbox;
    TKalmanFilter kf;
    int age;
    int visible_count;
    int invisible_count;

    tracks(int id_, Rect bbox_, TKalmanFilter kf_);
    void setFixed(int n_, int m_, unsigned int type_);
    void update(const darknet_ros_msgs::BoundingBox& box);
    void predict(double dT);
    void init(const darknet_ros_msgs::BoundingBox& box); //initial state vectors;
    void updateDetection(Rect bbox_);
    void noDetection();
    Point getCenter();
    
};
