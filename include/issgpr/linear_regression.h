#pragma once

#include "incremental_linear_regression.h"

#include <Eigen/Core>

class LinearRegression : public IncrementalLinearRegression {
  public:
    // Should we take different hyperparameters for every output dimension?
    virtual void init(int input_dims, int output_dims, double sig_n);
    void save_to_file(const char *filename) const override;

  protected:
    Eigen::VectorXd map(const Eigen::VectorXd& input) const override;
    Eigen::MatrixXd mapd(const Eigen::VectorXd& input) const override;
    std::vector<Eigen::MatrixXd> mapdd(const Eigen::VectorXd& input) const override;
};
