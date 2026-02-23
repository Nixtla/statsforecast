#define _USE_MATH_DEFINES
#include <cmath>
#include <stdexcept>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace garch {
namespace py = pybind11;
using Eigen::VectorXd;

VectorXd compute_sigma2(const Eigen::Ref<const VectorXd> &x0,
                        const Eigen::Ref<const VectorXd> &x, int p, int q) {
  if (x0.size() < 1 + p + q) {
    throw std::invalid_argument(
        "compute_sigma2: x0 must have at least 1 + p + q elements");
  }
  if (x.size() == 0) {
    throw std::invalid_argument("compute_sigma2: x must be non-empty");
  }
  double w = x0[0];
  VectorXd alpha = x0.segment(1, p);
  VectorXd beta = x0.segment(p + 1, q);
  VectorXd alpha_rev = alpha.reverse();
  VectorXd beta_rev = beta.reverse();

  Eigen::Index n = x.size();
  double unconditional_var = (x.array() - x.mean()).square().mean();
  VectorXd sigma2 = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  // Initialize all slots up to max(p,q) with unconditional variance so that
  // the beta sum in early iterations does not skip NaN entries silently.
  int start = std::max(p, q);
  for (int i = 0; i < start && i < n; ++i) {
    sigma2[i] = unconditional_var;
  }
  for (Eigen::Index k = start; k < n; ++k) {
    double psum = 0.0;
    for (int j = 0; j < p; ++j) {
      double val = x[k - p + j];
      psum += alpha_rev[j] * val * val;
    }
    double result = w + psum;
    if (q != 0) {
      double qsum = 0.0;
      for (int j = 0; j < q; ++j) {
        double sv = sigma2[k - q + j];
        if (!std::isnan(sv)) {
          qsum += beta_rev[j] * sv;
        }
      }
      result += qsum;
    }
    sigma2[k] = result;
  }
  return sigma2;
}

double loglik(const VectorXd &x0, const VectorXd &x, int p, int q) {
  VectorXd sigma2 = compute_sigma2(x0, x, p, q);

  // z = x - nanmean(x)
  double mean_x = 0.0;
  int count = 0;
  for (Eigen::Index i = 0; i < x.size(); ++i) {
    if (!std::isnan(x[i])) {
      mean_x += x[i];
      count++;
    }
  }
  if (count == 0) {
    return std::numeric_limits<double>::infinity();
  }
  mean_x /= count;

  double ll = 0.0;
  int start = std::max(p, q);
  for (Eigen::Index k = start; k < x.size(); ++k) {
    double s2 = sigma2[k];
    if (s2 == 0.0) {
      s2 = 1e-10;
    }
    double z = x[k] - mean_x;
    ll -= 0.5 * (std::log(2.0 * M_PI) + std::log(s2) + (z * z) / s2);
  }
  return -ll; // negative log-likelihood for minimization
}

double constraint_value(const VectorXd &x0) {
  // alpha + beta < 1 constraint: returns 1 - sum(x0[1:])
  return 1.0 - x0.tail(x0.size() - 1).sum();
}

void init(py::module_ &m) {
  py::module_ garch_mod = m.def_submodule("garch");
  garch_mod.def("compute_sigma2", &compute_sigma2,
                py::call_guard<py::gil_scoped_release>());
  garch_mod.def("loglik", &loglik,
                py::call_guard<py::gil_scoped_release>());
  garch_mod.def("constraint_value", &constraint_value,
                py::call_guard<py::gil_scoped_release>());
}

} // namespace garch
