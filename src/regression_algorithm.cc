#include "regression_algorithm.h"

#include <iostream>

void RegressionAlgorithm::init(int input_dims, int output_dims) {
  input_dims_ = input_dims;
  output_dims_ = output_dims;
}

void RegressionAlgorithm::save_to_file(const char* filename) const {
  std::cerr << "save to file not overridden! no saving happening." << std::endl;
}
