#include "issgpr.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

void IssgprModel::init(int input_dims, int output_dims,
                  int D, double sig_n, double sig_f,
                  const std::vector<double>& length_scales) {
  IncrementalLinearRegression::init(input_dims, output_dims, 2 * D, sig_n);

  if (length_scales.size() != static_cast<unsigned int>(input_dims)) {
    throw std::invalid_argument("ISSGPR: Length scales not the same size as input_dims");
  }

  D_ = D;
  sig_n_ = sig_n;
  sig_f_ = sig_f;

  // TODO Seed the generator with the time, perhaps?
  // Currently, the randomness is the same every time.
  std::default_random_engine generator;
  std::normal_distribution<double> dist;

  // Randomly pick features.
  Omega_.resize(D, input_dims_);
  for (int i = 0; i < D; i++) {
    for (int j = 0; j < input_dims_; j++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
      Omega_(i, j) = dist(generator) / length_scales[j];
#pragma GCC diagnostic pop
    }
  }
}

Eigen::VectorXd IssgprModel::map(const Eigen::VectorXd& input) const {
  Eigen::VectorXd trig_input = Omega_ * input;

  Eigen::VectorXd phi(2 * D_);
  phi << Eigen::cos(trig_input.array()), Eigen::sin(trig_input.array());
  phi *= sig_f_ / std::sqrt(D_);

  return phi;
}

Eigen::MatrixXd IssgprModel::mapd(const Eigen::VectorXd& input) const {
  Eigen::VectorXd trig_input = Omega_ * input;

  Eigen::ArrayXd ss = Eigen::sin(trig_input.array());
  Eigen::ArrayXd cs = Eigen::cos(trig_input.array());

  Eigen::MatrixXd deriv(2 * D_, input_dims_);
  for (int i = 0; i < input_dims_; i++) {
    deriv.block(0, i, D_, 1) = -ss * Omega_.col(i).array();
    deriv.block(D_, i, D_, 1) = cs * Omega_.col(i).array();
  }

  return (sig_f_ / std::sqrt(D_)) * deriv;
}

std::vector<Eigen::MatrixXd> IssgprModel::mapdd(const Eigen::VectorXd& input) const {
  Eigen::VectorXd trig_input = Omega_ * input;

  Eigen::ArrayXd cs = Eigen::cos(trig_input.array());
  Eigen::ArrayXd ss = Eigen::sin(trig_input.array());

  std::vector<Eigen::MatrixXd> dderiv(input_dims_, Eigen::MatrixXd(2 * D_, input_dims_));
  for (int i = 0; i < input_dims_; i++) {
    for (int j = 0; j < input_dims_; j++) {
      Eigen::ArrayXd mult = Omega_.col(i).array() * Omega_.col(j).array();
      dderiv[i].block(0, j, D_, 1) = -cs * mult;
      dderiv[i].block(D_, j, D_, 1) = -ss * mult;
    }

    dderiv[i] *= sig_f_ / std::sqrt(D_);
  }

  return dderiv;
}

void IssgprModel::save_to_file(const char *filename) const {
  std::ofstream output_file(filename);
  Eigen::IOFormat Fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
  output_file << input_dims_ << std::endl;
  output_file << D_ << std::endl;
  output_file << Omega_.format(Fmt) << std::endl;
  output_file << sig_n_ << std::endl;
  output_file << sig_f_ << std::endl;
  output_file << output_dims_ << std::endl;
  for (int i = 0; i < output_dims_; i++) {
    output_file << w_[i].format(Fmt) << std::endl;
  }

  output_file.close();
}
