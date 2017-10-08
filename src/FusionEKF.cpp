#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {

  /**
    * Initialize the FusionEKF.
    * Set the process and measurement noises
    */

  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  x_ = VectorXd(4);
  P_ = MatrixXd(4, 4);
  F_ = MatrixXd(4, 4);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  Q_ = MatrixXd(4, 4);

  // initiate state vector x_
  x_ << 1, 1, 1, 1;

  //the initial state covariance matrix P_
  P_ << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1000, 0,
            0, 0, 0, 1000;

  //the initial transition matrix F_
  F_ << 1, 0, 1, 0,
			  0, 1, 0, 1,
			  0, 0, 1, 0,
			  0, 0, 0, 1;

  //
  H_laser_ << 1, 0, 0, 0,
            0, 1, 0, 0;

  Hj_ << 1, 1, 0, 0,
         1, 1, 0, 0,
         1, 1, 1, 1;

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;


  const float noise_ax = 9;
  const float noise_ay = 9;

  //set initial process covariance matrix Q_
  Q_ << 0, 0, 0, 0,
			  0, 0, 0, 0,
			  0, 0, 0, 0,
			  0, 0, 0, 0;

}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {
    /**
      * Initialize the state ekf_.x_ with the first measurement.
      */

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
        * Convert radar from polar to cartesian coordinates and initialize state.
        * Calculate Jacobian Matrix.
        * Initialize state.
      */
      x_ << measurement_pack.raw_measurements_[0]*cos(measurement_pack.raw_measurements_[1]),
                measurement_pack.raw_measurements_[0]*sin(measurement_pack.raw_measurements_[1]),
            measurement_pack.raw_measurements_[2]*cos(measurement_pack.raw_measurements_[1]),
                measurement_pack.raw_measurements_[2]*sin(measurement_pack.raw_measurements_[1]);

      Hj_ = tools.CalculateJacobian(x_);

      ekf_.Init(x_, P_, F_, Hj_, R_radar_, Q_);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
          Initialize state.
      */
      x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 1, 1;
      ekf_.Init(x_, P_, F_, H_laser_, R_laser_, Q_);
    }

    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  /**
    * Update the state transition matrix F according to the new elapsed time.
      - Time is measured in seconds.
    * Update the process noise covariance matrix.
    * Use noise_ax = 9 and noise_ay = 9 for your Q matrix.
   */
  float d_time = (measurement_pack.timestamp_ - previous_timestamp_)/1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  ekf_.F_ << 1, 0, d_time, 0,
            0, 1, 0, d_time,
            0, 0, 1, 0,
            0, 0, 0, 1;
  ekf_.Q_ << tools.CalculateQ(noise_ax, noise_ay, d_time);
  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
    * Use the sensor type to perform the update step.
    * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  previous_timestamp_ = measurement_pack.timestamp_;
  // print the output
  //cout << "x_ = " << ekf_.x_ << endl;
  //cout << "P_ = " << ekf_.P_ << endl;
}
