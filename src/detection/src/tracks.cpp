#include "detection/tracks.h"

using namespace std;
using namespace cv;

tracks::tracks(int id_, Rect bbox_, int n_, int m_, unsigned int type_) {
  id = id_;
  bbox = bbox_;
  TKalmanFilter kf(n_, m_);
  age = 0;
  visible_count = 0;
  invisible_count = 0;

}

void tracks::setFixed(int n_, int m_, unsigned int type_) {
  kf.setFixed(n_, m_, type_);
}

void tracks::init(const darknet_ros_msgs::BoundingBox& box) {
  kf.init(box);
}

// might change with addition of hungarian algo
void tracks::update(const darknet_ros_msgs::BoundingBox& box) {
  age += 1;
  visible_count += 1;
  kf.update(box);
}

void tracks::predict(double dT) {
  kf.predict(dT);
}

void tracks::updateDetection(Rect bbox_) {
  bbox = bbox_;
}

void tracks::noDetection() {
  invisible_count += 1;
}

/**
 * Returns the center of the predicted point and also shifts the bbox 
 * so that its center is at the predicted location.
 */
Point tracks::getCenter() {
  Point center = kf.getCenter();

  bbox.x = center.x;
  bbox.y = center.y;

  return center;
}
