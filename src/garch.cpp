#include <cmath>
#include <random>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "nelder_mead.h"

namespace garch {
namespace py = pybind11;
using Eigen::VectorXd;

VectorXd compute_sigma2(const Eigen::Ref<const VectorXd> &x0,
                        const Eigen::Ref<const VectorXd> &x, int p, int q) {
  double w = x0[0];
  VectorXd alpha = x0.segment(1, p);
  VectorXd beta = x0.segment(p + 1, q);
  VectorXd alpha_rev = alpha.reverse();
  VectorXd beta_rev = beta.reverse();

  Eigen::Index n = x.size();
  VectorXd sigma2 = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  sigma2[0] = (x.array() - x.mean()).square().mean(); // unconditional variance

  int start = std::max(p, q);
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

// Penalized objective for Nelder-Mead: loglik + penalty if constraint violated
double penalized_loglik(const VectorXd &x0, const VectorXd &x, int p, int q) {
  // Check non-negativity
  if ((x0.array() < 0.0).any()) {
    return 1e20;
  }
  // Check constraint: sum of alpha + beta < 1
  if (constraint_value(x0) < 0.0) {
    return 1e20;
  }
  return loglik(x0, x, p, q);
}

// Full model fitting using Nelder-Mead
std::tuple<VectorXd, VectorXd, VectorXd>
fit(const Eigen::Ref<const VectorXd> &x, int p, int q) {
  int n_params = p + q + 1;
  VectorXd x0 = VectorXd::Constant(n_params, 0.1);
  VectorXd lower = VectorXd::Constant(n_params, 1e-8);
  VectorXd upper = VectorXd::Constant(n_params, 0.999);

  auto result = nm::NelderMead(penalized_loglik, x0, lower, upper,
                               0.05, 1e-4, 1.0, 2.0, 0.5, 0.5, 2000, 1e-10,
                               true, x, p, q);

  VectorXd coeff = std::get<0>(result);
  VectorXd sigma2 = compute_sigma2(coeff, x, p, q);

  // Generate fitted values using deterministic RNG (seed=1)
  std::mt19937 gen(1);
  std::normal_distribution<double> dist(0.0, 1.0);
  VectorXd fitted = VectorXd::Constant(x.size(), std::numeric_limits<double>::quiet_NaN());
  for (Eigen::Index k = p; k < x.size(); ++k) {
    double error = dist(gen);
    fitted[k] = error * std::sqrt(sigma2[k]);
  }

  return {coeff, sigma2, fitted};
}

// Forecast
std::tuple<VectorXd, VectorXd>
forecast(const Eigen::Ref<const VectorXd> &coeff, int p, int q, int h,
         const Eigen::Ref<const VectorXd> &y_vals_in,
         const Eigen::Ref<const VectorXd> &sigma2_vals_in) {
  double w = coeff[0];
  VectorXd alpha = coeff.segment(1, p);
  VectorXd beta = coeff.segment(p + 1, q);

  VectorXd y_vals = VectorXd::Constant(h + p, std::numeric_limits<double>::quiet_NaN());
  VectorXd sigma2_vals = VectorXd::Constant(h + q, std::numeric_limits<double>::quiet_NaN());

  y_vals.head(p) = y_vals_in;
  if (q != 0) {
    sigma2_vals.head(q) = sigma2_vals_in;
  }

  std::mt19937 gen(1);
  std::normal_distribution<double> dist(0.0, 1.0);

  VectorXd alpha_rev = alpha.reverse();
  VectorXd beta_rev = beta.reverse();

  for (int k = 0; k < h; ++k) {
    double error = dist(gen);
    double psum = 0.0;
    for (int j = 0; j < p; ++j) {
      double val = y_vals[k + j];
      if (!std::isnan(val)) {
        psum += alpha_rev[j] * val * val;
      }
    }
    double sigma2hat = w + psum;
    if (q != 0) {
      double qsum = 0.0;
      for (int j = 0; j < q; ++j) {
        double sv = sigma2_vals[k + j];
        if (!std::isnan(sv)) {
          qsum += beta_rev[j] * sv;
        }
      }
      sigma2hat += qsum;
    }
    double yhat = error * std::sqrt(sigma2hat);
    y_vals[p + k] = yhat;
    sigma2_vals[q + k] = sigma2hat;
  }

  return {y_vals.tail(h), sigma2_vals.tail(h)};
}

VectorXd generate_data(int n, double w,
                       const Eigen::Ref<const VectorXd> &alpha,
                       const Eigen::Ref<const VectorXd> &beta) {
  int p = alpha.size();
  int q = beta.size();

  if (w < 0 || (alpha.array() < 0).any() || (beta.array() < 0).any()) {
    throw std::invalid_argument("Coefficients must be nonnegative");
  }
  if (alpha.sum() + beta.sum() >= 1.0) {
    throw std::invalid_argument(
        "Sum of coefficients of lagged versions of the series and lagged "
        "versions of volatility must be less than 1");
  }

  VectorXd y = VectorXd::Zero(n);
  VectorXd sigma2 = VectorXd::Zero(n);

  std::mt19937 gen(1);
  std::normal_distribution<double> dist(0.0, 1.0);

  if (q != 0) {
    for (int i = 0; i < q; ++i) {
      sigma2[i] = 1.0;
    }
  }
  for (int k = 0; k < p; ++k) {
    y[k] = dist(gen);
  }

  VectorXd alpha_rev = alpha.reverse();
  VectorXd beta_rev = beta.reverse();

  int start = std::max(p, q);
  for (int k = start; k < n; ++k) {
    double psum = 0.0;
    for (int j = 0; j < p; ++j) {
      double val = y[k - p + j];
      psum += alpha_rev[j] * val * val;
    }
    double result = w + psum;
    if (q != 0) {
      double qsum = 0.0;
      for (int j = 0; j < q; ++j) {
        qsum += beta_rev[j] * sigma2[k - q + j];
      }
      result += qsum;
    }
    sigma2[k] = result;
    y[k] = dist(gen) * std::sqrt(sigma2[k]);
  }
  return y;
}

void init(py::module_ &m) {
  py::module_ garch_mod = m.def_submodule("garch");
  garch_mod.def("compute_sigma2", &compute_sigma2);
  garch_mod.def("loglik", &loglik);
  garch_mod.def("constraint_value", &constraint_value);
  garch_mod.def("fit", &fit);
  garch_mod.def("forecast", &forecast);
  garch_mod.def("generate_data", &generate_data);
}

} // namespace garch
