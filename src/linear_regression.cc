#include "linear_regression.h"

#include <fstream>

void LinearRegression::init(int input_dims, int output_dims, double sig_n) {
  IncrementalLinearRegression::init(input_dims, output_dims, input_dims, sig_n);
}

Eigen::VectorXd LinearRegression::map(const Eigen::VectorXd& input) const {
  return input;
}

Eigen::MatrixXd LinearRegression::mapd(const Eigen::VectorXd&) const {
  return Eigen::MatrixXd::Identity(input_dims_, input_dims_);
}

std::vector<Eigen::MatrixXd> LinearRegression::mapdd(const Eigen::VectorXd&) const {
  std::vector<Eigen::MatrixXd> dderiv(input_dims_);
  for (int i = 0; i < input_dims_; i++) {
    dderiv[i] = Eigen::MatrixXd::Zero(input_dims_, input_dims_);
  }

  return dderiv;
}

void LinearRegression::save_to_file(const char *filename) const {
  std::ofstream output_file(filename);
  Eigen::IOFormat Fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  output_file << input_dims_ << std::endl;
  output_file << output_dims_ << std::endl;
  for (int i = 0; i < output_dims_; i++) {
    output_file << w_[i].format(Fmt) << std::endl;
  }

  output_file.close();
}
