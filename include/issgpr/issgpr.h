#pragma once

#include "incremental_linear_regression.h"

#include <utility>
#include <vector>

#include <Eigen/Core>

class IssgprModel : public IncrementalLinearRegression {
  public:
    // Should we take different hyperparameters for every output dimension?
    virtual void init(int input_dims, int output_dims,
         int D, double sig_n, double sig_f,
         const std::vector<double>& length_scales);

    void save_to_file(const char *filename) const override;

  protected:
    Eigen::VectorXd map(const Eigen::VectorXd& input) const override;
    Eigen::MatrixXd mapd(const Eigen::VectorXd& input) const override;
    std::vector<Eigen::MatrixXd> mapdd(const Eigen::VectorXd& input) const override;

    // Random features (frequencies).
    Eigen::MatrixXd Omega_;

    // The number of random features to use.
    int D_;

    // The process noise (assumed stddev of noise added to training inputs).
    double sig_n_;
    // The signal stddev.
    double sig_f_;
};
