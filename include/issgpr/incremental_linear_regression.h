#pragma once

#include "regression_algorithm.h"

#include <utility>
#include <vector>

#include <Eigen/Cholesky>

class IncrementalLinearRegression : public RegressionAlgorithm {
  public:
    // Should we take different hyperparameters for every output dimension?
    virtual void init(int input_dims, int output_dims, int feature_dims, double sig_n);

    // Returns the predicted value for the input before the update.
    virtual Eigen::VectorXd update(const Eigen::VectorXd& input,
                                   const Eigen::VectorXd& output) override;

    // Returns the predicted value for the input.
    virtual Eigen::VectorXd predict(const Eigen::VectorXd& input) const override;

    Eigen::MatrixXd get_deriv(const Eigen::VectorXd& input) const;
    std::vector<Eigen::MatrixXd> get_dderiv(const Eigen::VectorXd& input) const;

  protected:
    virtual Eigen::VectorXd map(const Eigen::VectorXd& input) const = 0;
    virtual Eigen::MatrixXd mapd(const Eigen::VectorXd& input) const = 0;
    virtual std::vector<Eigen::MatrixXd> mapdd(const Eigen::VectorXd& input) const = 0;

    // Regression parameters: Phi(X) * w = y. (one for each output dimension).
    std::vector<Eigen::VectorXd> w_;

  private:
    virtual std::pair<Eigen::VectorXd, Eigen::VectorXd> map_and_predict(const Eigen::VectorXd& input) const;

    // Cholesky decomposition of Phi(X).T * Phi(X) + sig_n * I
    Eigen::LLT<Eigen::MatrixXd, Eigen::Upper> A_;
    // Phi(X).T * y (one for each output dimension).
    std::vector<Eigen::VectorXd> b_;
};
