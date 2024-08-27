#pragma once

#include <pybind11/eigen.h>

struct OptimResult {
  Eigen::VectorXd x;
  double fun;
  int nit;
};
