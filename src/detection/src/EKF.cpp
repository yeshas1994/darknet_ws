#include "EKF.h"

using namespace Eigen

EKF::ExtKalmanFilter() {}

/**
 * For ekf F_ (transition) & H_ (measurement) matrices are Jacobians 
 * of their respective transfer functions
 */
void EKF::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
             MatrixXd &Q_in, MatrixXd &H_in, MatrixXd &R_in) {

  x_ = x_in; // state vector 
  P_ = P_in; // state covariance matrix
  F_ = F_in; // state transition
  Q_ = Q_in; // process covariance
  H_ = H_in; // measurement matrix
  R_ = R_in; // measurement covariance
  missing_count = 0;
}

void EKF::Predict(double dT) {

  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;

}

void EKF::Update(const VectorXd &z) {

  VectorXd z_pred = H_ * x_;

  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * (P_ * Ht) + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si; //kalman gain

  x_ = x_ + (K * y);
  long mat_size = x_.size();
  MatrixXd I_ = MatrixXd::Identity(x_size, x_size);
  P_ = ( I_ - (K * H_) ) * P_;

}

void EKF::noDetection() {

  missing_count++;

} 

cv::Point getCentroid() {

  return cv::Point(x_(0), x_(1));

}

cv::Rect getBox() {

  return cv::Rect( (x_(0) - x_(4) / 2), (x_(1) - x_(5) / 2),
                   (x_(0) + x_(4) / 2), (x_(1) + x_(5) / 2) );

}
