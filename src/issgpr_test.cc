#include "issgpr.h"

#include <chrono>
#include <iostream>

constexpr int input_dim = 2;
constexpr int output_dim = 2;
constexpr int D = 100;

typedef Eigen::Matrix<double, input_dim, 1> input_t;
typedef Eigen::Matrix<double, output_dim, 1> output_t;

int main(int argc, char **argv) {
  IssgprModel model;
  model.init(input_dim, output_dim, D, 0.1, 0.1, {0.14, 0.3});
  input_t input(1, 0);
  output_t output(5, 2);

  constexpr int N = 10000;
  double avg_us = 0;
  for (int i = 0; i < N; i++) {
    auto start = std::chrono::steady_clock::now();

    auto p = model.update(input, output);
    input[0] += 0.04;

    auto end = std::chrono::steady_clock::now();
    avg_us += std::chrono::duration<double, std::micro>(end - start).count() / N;
  }

  std::cout << "ISSGPR with " << D << " random features." << std::endl;
  std::cout << "Avg update time: " << avg_us << " microseconds" << std::endl;

  std::cout << "Checking 1st and 2nd derivatives..." << std::endl;

  // Perturb a bit so gradient is non zero...?
  input[0] += 0.1;

  constexpr double eps = 1e-9;
  Eigen::Matrix<double, input_dim, input_dim> dnum;

  auto val = model.predict(input);
  for (int i = 0; i < input_dim; i++) {
    auto input2 = input;
    input2[i] += eps;
    dnum.col(i) = (model.predict(input2) - val) / eps;
  }

  std::cout << "D1 Ana:" << std::endl << model.get_deriv(input) << std::endl;
  std::cout << "D1 Num:" << std::endl << dnum << std::endl;

  std::cout << "D2 Ana:" << std::endl;
  auto dd = model.get_dderiv(input);
  for (int i = 0; i < input_dim; i++) {
    std::cout << dd[i] << std::endl;
  }

  constexpr double eps2 = 1e-5;
  std::cout << "D2 Num:" << std::endl;
  for (int i = 0; i < input_dim; i++) {
    input_t input2 = input;
    input2[i] += eps2;
    output_t d1 = (model.predict(input2) - val) / eps2;

    Eigen::Matrix<double, input_dim, input_dim> dnum2;

    for (int j = 0; j < input_dim; j++) {
      input_t input3 = input;
      input3[j] += eps2;
      input_t input4 = input2;
      input4[j] += eps2;
      output_t d2 = (model.predict(input4) - model.predict(input3)) / eps2;
      dnum2.col(j) = (d2 - d1) / eps2;
    }

    std::cout << dnum2 << std::endl;
  }
}
