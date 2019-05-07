#include <Eigen/Dense>

class ExtKalmanFilter {

public:
  //state vector 
  Eigen::VectorXd x_;
  
  //state covariance matrix
  Eigen::MatrixXd P_;

  //state transition matrix
  Eigen::MatrixXd F_;

  //process covariance matrix
  Eigen::MatrixXd Q_;

  //state measurement matrix
  Eigen::MatrixXd H_;

  //measurement covariance matrix
  Eigen::MatrixXd R_;

  ExtKalmanFilter();

  // Initializes Kalman Filter
  void Init(Eigen::VectorXd &x_in, Eigen::MatrixXd &P_in, Eigen::MatrixXd &F_in,
            Eigen::MatrixXd &Q_in, Eigen::MatrixXd &H_in, Eigen::MatrixXd &R_in);

  // Predicts the state and state covariance after time delta T using the
  // process model
  void Predict(double dT);

  // Updates the state by using measured values 
  // Normal Kalman Filter
  // z - measurement vector
  void Update(const Eigen::VectorXd &z);

  // Updates the state by using measured values 
  // Extended Kalman Filter
  // z - measurement vector
  void UpdateEKF(const Eigen::VectorXd &z);

};
