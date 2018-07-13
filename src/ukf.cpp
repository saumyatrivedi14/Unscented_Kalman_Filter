#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
 
UKF::UKF() {
	
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 1;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // True when first ProcessMeasurement called
  is_initialized_ = false;
  
  // start time in us
  time_us_ = 0.0;
  
  // number of states = 5
  n_x_ = 5;
  
  // number of augmented states = 7
  n_aug_ = n_x_ + 2;
  
  // number of sigma points = 15
  n_sig_ = 2*n_aug_+1;
  
  // sigma points spreading parameter
  lambda_ = 3 - n_x_;
  
  // weights vector
  weights_ = VectorXd(n_sig_);
  
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);
  
  // init Radar Measurement Noise Matrix
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
			0, std_radphi_*std_radphi_, 0,
			0, 0, std_radrd_*std_radrd_;
			
  // init Ladar Measurement Noise Matrix
  R_lidar_ = MatrixXd(2,2);
  R_lidar_ << std_laspx_*std_laspx_, 0,
			0, std_laspy_*std_laspy_;
  
  // NIS values for Laser & Radar
  NIS_L_ = 0.0;
  NIS_R_ = 0.0;  
  
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (is_initialized_){
		
	
		// init covariance matrix
		P_ = MatrixXd::Identity(5, 5);
		
		// init weights
		for (int i=0; i<n_sig_; i++){
		  if (i == 0){
			  weights_(i) = lambda_/(lambda_+n_aug_);
		  }else{
			  weights_(i) = 0.5/(lambda_+n_aug_);
		  }
		}
		
		// init time-stamp
		time_us_ = meas_package.timestamp_;
		
		if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
			float rho = meas_package.raw_measurements_(0);
			float phi = meas_package.raw_measurements_(1);
			float rho_dot = meas_package.raw_measurements_(2);

			float px = rho*cos(phi);
			float py = rho*sin(phi);
			float vx = rho_dot*cos(phi);
			float vy = rho_dot*sin(phi);
			float v = sqrt(pow(vx,2) + pow(vy,2));
			
			x_ << px, py, v, 0.0, 0.0;
						
		}
		else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
			float px = meas_package.raw_measurements_(0);
			float py = meas_package.raw_measurements_(1);
			
			x_ << px, py, 0.0, 0.0, 0.0;
		}
		
	
		// done initializing
		is_initialized_ = true;
		return;
}

/**************************
	PREDICTION
***************************/
	double delta_t = (meas_package.timestamp_ - time_us_)/1000000;
	time_us_ = meas_package.timestamp_;

	Prediction(delta_t);
	
/**************************
	UPDATE
***************************/

	if ((meas_package.sensor_type_ == MeasurementPackage::RADAR) && (use_radar_ == true)){
		UpdateRadar(meas_package);
	}
	else if ((meas_package.sensor_type_ == MeasurementPackage::LASER) && (use_laser_ == true)){
		UpdateLidar(meas_package);
	}
	
	// print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}
	
/**************************
	IMP FUNCTIONS
***************************/
	
void UKF::Prediction(double delta_t){
  
  /****	Generating Sigma Points ****/
  
  //create sigma points matrix
  MatrixXd Xsig = MatrixXd(n_x_, n_sig_);
  
  //create square root matrix of P
  MatrixXd A = P_.llt().matrixL();
  
  Xsig.col(0) = x_;
  
  for (int i=0; i<n_x_; i++){
    Xsig.col(i+1) = x_ + sqrt(lambda_+n_x_)*A.col(i);
    
    Xsig.col(i+n_x_+1) = x_ - sqrt(lambda_+n_x_)*A.col(i);
  }
  
  /****	Augmented Sigma Points ****/
  
  //create augmented mean state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0, 0;

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug << MatrixXd::Zero(n_aug_, n_aug_);

  //create augmented sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  //create augmented covariance matrix
  MatrixXd Q = MatrixXd(2,2);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;
	   
  P_aug.topLeftCorner(5,5) = P_;
  P_aug.bottomRightCorner(2,2) = Q;
  
  //create square root matrix of P_aug
  MatrixXd A_aug = P_aug.llt().matrixL();
 
  //create augmented sigma points
  Xsig_aug.col(0) = x_aug;
  for (int i = 0; i< n_aug_; i++){
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * A_aug.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A_aug.col(i);
  }
  
  /****	Predict Augmented Sigma Points ****/
  
  for(int i=0; i<n_sig_; i++){
      float px       = Xsig_aug(0,i);
      float py       = Xsig_aug(1,i);
      float v        = Xsig_aug(2,i);
      float yaw      = Xsig_aug(3,i);
      float yaw_dot  = Xsig_aug(4,i);
      float nu_a     = Xsig_aug(5,i);
      float nu_d2yaw = Xsig_aug(6,i);
      
      //initializing predicted states
      float px_p, py_p, v_p, yaw_p, yaw_dot_p;
      
      //avoiding division by zero case
      if(yaw_dot < 0.0001){
          px_p = px + (v * delta_t * cos(yaw));
          py_p = py + (v * delta_t * sin(yaw));
      }else{
          px_p = px + ((v/yaw_dot) * (sin(yaw + yaw_dot * delta_t) - sin(yaw)));
          py_p = py + ((v/yaw_dot) * (-cos(yaw + yaw_dot * delta_t) + cos(yaw)));
      }
      
      //adding process noise
      px_p +=  0.5 * pow(delta_t,2) * cos(yaw) * nu_a; 
      py_p +=  0.5 * pow(delta_t,2) * sin(yaw) * nu_a;

      v_p = v + (delta_t * nu_a);
      yaw_p = yaw + (yaw_dot * delta_t) + (0.5 * pow(delta_t,2) * nu_d2yaw);
      yaw_dot_p = yaw_dot + (delta_t * nu_d2yaw);
      
      Xsig_pred_(0,i) = px_p;
      Xsig_pred_(1,i) = py_p;
      Xsig_pred_(2,i) = v_p;
      Xsig_pred_(3,i) = yaw_p;
      Xsig_pred_(4,i) = yaw_dot_p;
	}
	
  /**** Predict Mean and Covariance Matrix ****/
  
  // set predicted mean and covariance to zero
  VectorXd x_pred = VectorXd(n_x_);
  x_pred.fill(0.0);
  MatrixXd P_pred = MatrixXd(n_x_, n_x_);
  P_pred.fill(0.0);

  for (int i=0; i<n_sig_; i++){
      x_pred += weights_(i) * Xsig_pred_.col(i);
  }
  
  for (int i=0; i<n_sig_; i++){
      VectorXd diff_x = Xsig_pred_.col(i) - x_;
      
	  // normalizing
      while (diff_x(3) > M_PI) {diff_x(3) -= 2*M_PI;}
      while (diff_x(3) < -M_PI) {diff_x(3) += 2*M_PI;}
      
      P_pred += weights_(i) * diff_x * diff_x.transpose();
  }
  
  x_ = x_pred;
  P_ = P_pred;
  
}

void UKF::UpdateLidar(MeasurementPackage meas_package){
  
  // dimension of lidar measurements
  int n_z = 2;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);
  
  
  for (int i=0; i<n_sig_; i++){
      //transform sigma points into measurement space
	  Zsig(0,i) = Xsig_pred_(0,i);
      Zsig(1,i) = Xsig_pred_(1,i);
	  
	  //calculate mean predicted measurement
	  z_pred += weights_(i) * Zsig.col(i);
	  
  }
  
  //calculate measurement covariance matrix S
  for (int i=0; i<n_sig_; i++){
      VectorXd diff_z = Zsig.col(i)-z_pred;
      
      S += weights_(i) * diff_z *diff_z.transpose();
  }

  //add measurement noise covariance matrix
  S += R_lidar_;
  
  //vector for incoming lidar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),   //px in m
	   meas_package.raw_measurements_(1);   //py in m
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i<n_sig_; i++){
      VectorXd diff_x = Xsig_pred_.col(i) - x_;
      VectorXd diff_z = Zsig.col(i) - z_pred;
      
      while(diff_x(3) > M_PI){diff_x(3) -= 2*M_PI;}
      while(diff_x(3) < -M_PI){diff_x(3) += 2*M_PI;}
      
      Tc += weights_(i)*diff_x*diff_z.transpose();
  }
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  VectorXd diff_z_err = z - z_pred;
  
  x_ += K*diff_z_err;
  P_ -= K * S * K.transpose();
  
  //NIS for lidar
  NIS_L_ = diff_z_err.transpose() * S.inverse() * diff_z_err;
}


void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //dimension of radar measurement (r, phi, and r_dot)
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  
  Zsig.fill(0.0);
  z_pred.fill(0.0);
  S.fill(0.0);
  
  //transform sigma points into measurement space
  for (int i=0; i<n_sig_; i++){
      float px  = Xsig_pred_(0,i);
      float py  = Xsig_pred_(1,i);
      float v   = Xsig_pred_(2,i);
      float yaw = Xsig_pred_(3,i);
      
      Zsig(0,i) = sqrt(pow(px,2)+pow(py,2));
      Zsig(1,i) = atan(py/px);
      Zsig(2,i) = (px*cos(yaw)*v + py*sin(yaw)*v)/sqrt(pow(px,2)+pow(py,2));
	  
	  //calculate mean predicted measurement
      z_pred += weights_(i) * Zsig.col(i);
  }
  
  //calculate innovation covariance matrix S
  for (int i=0; i<n_sig_; i++){
      VectorXd diff_z = Zsig.col(i)-z_pred;
      
	  // normalize angle
      while(diff_z(1) > M_PI){diff_z(1) -= 2*M_PI;}
      while(diff_z(1) < -M_PI){diff_z(1) += 2*M_PI;}
      
      S += weights_(i) * diff_z *diff_z.transpose();
  }

  S += R_radar_;
  
  // vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_(0),   //rho in m
       meas_package.raw_measurements_(1),   //phi in rad
       meas_package.raw_measurements_(2);   //rho_dot in m/s
	   
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i=0; i<n_sig_; i++){
      VectorXd diff_x = Xsig_pred_.col(i) - x_;
      VectorXd diff_z = Zsig.col(i) - z_pred;
      
	  //normalize angles
      while(diff_x(3) > M_PI){diff_x(3) -= 2*M_PI;}
      while(diff_x(3) < -M_PI){diff_x(3) += 2*M_PI;}
      while(diff_z(1) > M_PI){diff_z(1) -= 2*M_PI;}
      while(diff_z(1) > M_PI){diff_z(1) -= 2*M_PI;}
      
      Tc += weights_(i)*diff_x*diff_z.transpose();
  }
  
  //calculate Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  
  //update state mean and covariance matrix
  VectorXd diff_z_err = z - z_pred;
  
  //normalizing
  while(diff_z_err(1) > M_PI){diff_z_err(1) -= 2*M_PI;}
  while(diff_z_err(1) > M_PI){diff_z_err(1) -= 2*M_PI;}
  
  x_ += K*diff_z_err;
  P_ -= K * S * K.transpose();

  //NIS for radar
  NIS_R_ = diff_z_err.transpose() * S.inverse() * diff_z_err;
}