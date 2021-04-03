#pragma once

#include <Eigen/Core>

// Should we template or have run time number of dimensions?

class RegressionAlgorithm {
  public:
    RegressionAlgorithm() {}
    virtual void init(int input_dims, int output_dims);
    virtual ~RegressionAlgorithm() {}

    // Returns the predicted value for the input before the update.
    virtual Eigen::VectorXd update(const Eigen::VectorXd& input,
                                   const Eigen::VectorXd& output) = 0;

    // Returns the predicted value for the input.
    virtual Eigen::VectorXd predict(const Eigen::VectorXd& input) const = 0;

    virtual void save_to_file(const char *filename) const;

  protected:
    int input_dims_;
    int output_dims_;
};
