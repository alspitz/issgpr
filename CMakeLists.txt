cmake_minimum_required(VERSION 3.0.0)
project(issgpr)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED)
find_package(Eigen3 REQUIRED)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  DEPENDS EIGEN3
)

include_directories(include/${PROJECT_NAME} ${EIGEN3_INCLUDE_DIRS})

add_library(${PROJECT_NAME}
  src/issgpr.cc
  src/regression_algorithm.cc
  src/incremental_linear_regression.cc
  src/linear_regression.cc)

add_executable(issgpr_test src/issgpr_test.cc)
target_link_libraries(issgpr_test ${PROJECT_NAME})
