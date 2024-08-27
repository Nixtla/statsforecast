#pragma once

#include <pybind11/eigen.h>

namespace optim {
struct Result {
  Eigen::VectorXd x;
  double fun;
  int nit;
};
} // namespace optim
