#include "incremental_linear_regression.h"

#include <iostream>
#include <tuple>

#include <Eigen/Jacobi> // Had to include to get A_.rankUpdate to compile?

void IncrementalLinearRegression::init(int input_dims, int output_dims,
                                       int feature_dims, double sig_n) {
  RegressionAlgorithm::init(input_dims, output_dims);

  // XXX We initialize with sig_n^2 below because I'm not sure how
  // to initialize the Cholesky "object" with a factor
  // (the constructor factorizes) HAX TODO FIX.
  A_ = Eigen::LLT<Eigen::MatrixXd, Eigen::Upper>(sig_n * sig_n *
    Eigen::MatrixXd::Identity(feature_dims, feature_dims));

  for (int i = 0; i < output_dims_; i++) {
    b_.push_back(Eigen::MatrixXd::Zero(feature_dims, 1));
    w_.push_back(Eigen::MatrixXd::Zero(feature_dims, 1));
  }
}

Eigen::VectorXd IncrementalLinearRegression::predict(const Eigen::VectorXd& input) const {
  Eigen::VectorXd output;
  std::tie(output, std::ignore) = map_and_predict(input);
  return output;
}

Eigen::MatrixXd IncrementalLinearRegression::get_deriv(const Eigen::VectorXd& input) const {
  Eigen::MatrixXd phi_deriv = mapd(input);
  Eigen::MatrixXd deriv(output_dims_, input_dims_);
  for (int i = 0; i < output_dims_; i++) {
    deriv.row(i) = w_[i].transpose() * phi_deriv;
  }
  return deriv;
}

std::vector<Eigen::MatrixXd> IncrementalLinearRegression::get_dderiv(const Eigen::VectorXd& input) const {
  std::vector<Eigen::MatrixXd> phi_dderiv = mapdd(input);
  std::vector<Eigen::MatrixXd> dderiv(input_dims_);
  for (int i = 0; i < input_dims_; i++) {
    dderiv[i] = Eigen::MatrixXd(output_dims_, input_dims_);
    for (int j = 0; j < output_dims_; j++) {
      dderiv[i].row(j) = w_[j].transpose() * phi_dderiv[i];
    }
  }

  return dderiv;
}

Eigen::VectorXd IncrementalLinearRegression::update(const Eigen::VectorXd& input,
                                                    const Eigen::VectorXd& output) {
  Eigen::VectorXd predicted;
  Eigen::VectorXd phi;
  std::tie(predicted, phi) = map_and_predict(input);

  // Update the Cholesky decomposition and vector term.
  A_.rankUpdate(phi);
  for (int i = 0; i < output_dims_; i++) {
    b_[i] += phi * output(i);
  }

  // Update the linear regression parameters.
  for (int i = 0; i < output_dims_; i++) {
    w_[i] = A_.solve(b_[i]);
  }

  return predicted;
}

std::pair<Eigen::VectorXd, Eigen::VectorXd> IncrementalLinearRegression::map_and_predict(const Eigen::VectorXd& input) const {
  Eigen::VectorXd phi = map(input);
  Eigen::VectorXd output(output_dims_, 1);
  for (int i = 0; i < output_dims_; i++) {
    output(i) = w_[i].transpose() * phi;
  }

  return std::make_pair(output, phi);
}
