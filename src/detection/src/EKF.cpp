#include "EKF.h"

using namespace Eigen

EKF::ExtKalmanFilter() {}

/**
 * For ekf F_ (transition) & H_ (measurement) matrices are Jacobians 
 * of their respective transfer functions
 */
void EKF::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
             MatrixXd &Q_in, MatrixXd &H_in, MatrixXd &R_in) {

  x_ = x_in; 
  P_ = P_in;
  F_ = F_in;
  Q_ = Q_in;
  H_ = H_in;
  R_ = R_in;

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
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  long mat_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = ( I_ - (K * H) ) * P_;

}
