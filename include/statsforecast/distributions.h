#pragma once

#include <cmath>
#include <limits>
#include <numbers>

namespace dist {

enum class Distribution {
  Normal = 0,
  Laplace = 1,
  StudentT = 2,
  SkewNormal = 3,
  GED = 4,
};

// Number of distribution params appended to the optimizer vector.
// Laplace: 0 (scale b_hat is analytic). All others: 2.
inline int distribution_n_extra_params(Distribution d) {
  return (d == Distribution::Laplace) ? 0 : 2;
}

// Per-observation negative log-likelihood CORES.
// e: residual array (additive errors), length n. Returns +inf on degeneracy.

inline double negloglik_laplace(const double *e, int n) {
  double s = 0.0;
  for (int i = 0; i < n; ++i)
    s += std::abs(e[i]);
  double b_hat = s / static_cast<double>(n);
  if (b_hat <= 0.0)
    return std::numeric_limits<double>::infinity();
  return std::log(b_hat);
}

inline double negloglik_t(const double *e, int n, double log_sigma2,
                          double log_nu_m2) {
  double sigma2 = std::exp(log_sigma2);
  double nu = std::exp(log_nu_m2) + 2.0;
  double half_nu1 = 0.5 * (nu + 1.0);
  double sum_log_kernel = 0.0;
  for (int i = 0; i < n; ++i)
    sum_log_kernel += std::log(e[i] * e[i] / (nu * sigma2) + 1.0);
  return (0.5 * log_sigma2 + std::lgamma(nu / 2.0) - std::lgamma(half_nu1) +
          0.5 * std::log(nu * std::numbers::pi) +
          half_nu1 / static_cast<double>(n) * sum_log_kernel);
}

inline double negloglik_skewnorm(const double *e, int n, double log_sigma2,
                                 double alpha) {
  double sigma = std::exp(0.5 * log_sigma2);
  double sum_sq = 0.0;
  double sum_log_cdf = 0.0;
  for (int i = 0; i < n; ++i) {
    sum_sq += e[i] * e[i];
    double z = alpha * e[i] / (sigma * std::numbers::sqrt2);
    sum_log_cdf += std::log(0.5 * std::erfc(-z) + 1e-30);
  }
  return (-std::log(2.0) + 0.5 * std::log(2.0 * std::numbers::pi) +
          0.5 * log_sigma2 +
          sum_sq / (2.0 * static_cast<double>(n) * sigma * sigma) -
          sum_log_cdf / static_cast<double>(n));
}

inline double negloglik_ged(const double *e, int n, double log_sigma,
                            double log_beta) {
  double sigma = std::exp(log_sigma);
  double beta_ged = std::exp(log_beta);
  double sum_pow = 0.0;
  for (int i = 0; i < n; ++i)
    sum_pow += std::pow(std::abs(e[i]) / sigma, beta_ged);
  return (std::log(2.0) + log_sigma + std::lgamma(1.0 / beta_ged) - log_beta +
          sum_pow / static_cast<double>(n));
}

} // namespace dist
