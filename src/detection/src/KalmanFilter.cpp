#include "TKalmanFilter.hpp"

using namespace std;
using namespace cv;

TKalmanFilter::TKalmanFilter(int n_, int m_) {
    n = n_;
    m = m_;
    this->setFixed(n, m, type);
}

TKalmanFilter::init(const darknet_ros_msgs::BoundingBox::ConstPtr& box) {
    state(n, 1, type);
    meas(m, 1, type);

    meas.at<float>(0) = (box.xmin + box.xmax)/2.0;
    meas.at<float>(1) = (box.ymin + box.ymax)/2.0;
    meas.at<float>(2) = (float) (box.xmax - box.xmin);
    meas.at<float>(3) = (float) (box.ymax - box.ymin);

    kf_.errorCovPre.at<float>(0) = 1; // px
    kf_.errorCovPre.at<float>(7) = 1; // px
    kf_.errorCovPre.at<float>(14) = 1;
    kf_.errorCovPre.at<float>(21) = 1;
    kf_.errorCovPre.at<float>(28) = 1; // px
    kf_.errorCovPre.at<float>(35) = 1; // px

    state.at<float>(0) = meas.at<float>(0);
    state.at<float>(1) = meas.at<float>(1);
    state.at<float>(2) = 0;
    state.at<float>(3) = 0;
    state.at<float>(4) = meas.at<float>(2);
    state.at<float>(5) = meas.at<float>(3);

    kf_.statePost = state;
}

TKalmanFilter::setFixed(int n_, int m_, unsigned int type_) {
    kf_.statePost = Mat::zeros(n_, m_, type_);

    // Measure Matrix H
    // [1 0 0 0 0 0]
    // [0 1 0 0 0 0]
    // [0 0 0 0 1 0]
    // [0 0 0 0 0 1]
    kf_.measurementMatrix = Mat::zeros(m_, n_, type_);
    kf_.measurementMatrix.at<float>(0) = 1.0f;
    kf_.measurementMatrix.at<float>(7) = 1.0f;
    kf_.measurementMatrix.at<float>(16) = 1.0f;
    kf_.measurementMatrix.at<float>(23) = 1.0f;

    // Transition State Matrix A
    // Note: set dT while processing
    // dT added in estimation loop
    // [ 1 0 dT 0  0 0 ]
    // [ 0 1 0  dT 0 0 ]
    // [ 0 0 1  0  0 0 ]
    // [ 0 0 0  1  0 0 ]
    // [ 0 0 0  0  1 0 ]
    // [ 0 0 0  0  0 1 ]
    setIdentity(kf_.transitionMatrix);

    // Process Noise Covariance Matrix Q
    // [ Ex   0   0     0     0    0  ]
    // [ 0    Ey  0     0     0    0  ]
    // [ 0    0   Ev_x  0     0    0  ]
    // [ 0    0   0     Ev_y  0    0  ]
    // [ 0    0   0     0     Ew   0  ]
    // [ 0    0   0     0     0    Eh ]
    kf_.processNoiseCov.at<float>(0) = 1e-2;
    kf_.processNoiseCov.at<float>(7) = 1e-2;
    kf_.processNoiseCov.at<float>(14) = 1e-2;
    kf_.processNoiseCov.at<float>(21) = 1e-2;
    kf_.processNoiseCov.at<float>(28) = 1e-2;
    kf_.processNoiseCov.at<float>(35) = 1e-2;

    // Measures Noise Covariance Matrix R
    setIdentity(kf_.measurementNoiseCov, cv::Scalar(1e-5));
}

TKalmanFilter::predict() {
    double x_ =  kf_.statePost.at<float>(0);
    double y_ = kf_.statePost.at<float>(1);
    double dx_ = kf_.statePost.at<float>(2);
    double dy_ = kf_.statePost.at<float>(3);
    double w = kf_.statePost.at<float>(4);
    double h = kf_.statePost.at<float>(5);

    Mat prediction = kf_.predict();

    kf_.statePre.at<float>(0) = x_ + dx_ * dT;
    kf_.statePre.at<float>(1) = y_ + dy_ * dT;
    kf_.statePre.at<float>(2) = dx_;
    kf_.statePre.at<float>(3) = dy_;
    kf_.statePre.at<float>(4) = w; 
    kf_.statePre.at<float>(5) = h;
}

TKalmanFilter::update(const darknet_ros_msgs::BoundingBox::ConstPtr& box) {

    meas.at<float>(0) = (box.xmin + box.xmax)/2.0;
    meas.at<float>(1) = (box.ymin + box.ymax)/2.0;
    meas.at<float>(2) = (float) box.xmax - box.xmin;
    meas.at<float>(3) = (float) box.ymax - box.ymin;

    Mat estimate = kf_.correct(meas);
    
    kf_.temp5.at<float>(0) = meas.at<float>(0) - kf_.statePre.at<float>(0);
    kf_.temp5.at<float>(1) = meas.at<float>(1) - kf_.statePre.at<float>(1); 
    kf_.temp5.at<float>(2) = meas.at<float>(2) - kf_.statePre.at<float>(4); 
    kf_.temp5.at<float>(3) = meas.at<float>(3) - kf_.statePre.at<float>(5);
    kf_.statePost = kf_.statePre + kf_.gain * kf_.temp5;

}

TKalmanFilter::getCenter() {
    Point center;
    center.x = kf_.statePost.at<float>(0);
    center.y = kf_.statePost.at<float>(1);
    return center;
}
