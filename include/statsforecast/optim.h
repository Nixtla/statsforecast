#pragma once

#include <nanobind/eigen/dense.h>

struct OptimResult {
  Eigen::VectorXd x;
  double fun;
  int nit;
};
