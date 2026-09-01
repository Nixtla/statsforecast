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

// Safe box for the optimizer tail. The entries are logs, so an unbounded line
// search can overflow/underflow exp()/lgamma()/pow(): exp(-1008) == 0.0 then
// 1.0/beta. Python reads these through _lib.distributions.tail_bounds(), so the
// two sides cannot drift apart.
// Scale bounds are numeric-safety only and cannot bind for a real series; shape
// bounds also keep the fit usable by scipy's gennorm/t.
inline constexpr double kLogScaleMin = -250.0;  // exp -> [2.7e-109, 3.7e108]
inline constexpr double kLogScaleMax = 250.0;
inline constexpr double kLogNuM2Min = -15.0;  // log(nu-2): nu in (2, 1098.6]
inline constexpr double kLogNuM2Max = 7.0;
inline constexpr double kAlphaMin = -100.0;
inline constexpr double kAlphaMax = 100.0;
inline constexpr double kLogBetaMin = -3.0;  // log(beta): beta in [0.05, 50]
inline constexpr double kLogBetaMax = 3.912023005428146;

// (lo, hi) per tail entry; Normal/Laplace are unbounded.
struct TailBounds {
  double scale_lo, scale_hi, shape_lo, shape_hi;
};

inline constexpr TailBounds tail_bounds(Distribution d) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  switch (d) {
  case Distribution::StudentT:
    return {kLogScaleMin, kLogScaleMax, kLogNuM2Min, kLogNuM2Max};
  case Distribution::SkewNormal:
    return {kLogScaleMin, kLogScaleMax, kAlphaMin, kAlphaMax};
  case Distribution::GED:
    return {kLogScaleMin, kLogScaleMax, kLogBetaMin, kLogBetaMax};
  default:
    return {-kInf, kInf, -kInf, kInf};
  }
}

inline constexpr double Clamp(double x, double lo, double hi) {
  return x < lo ? lo : (x > hi ? hi : x);
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
  constexpr TailBounds b = tail_bounds(Distribution::StudentT);
  log_sigma2 = Clamp(log_sigma2, b.scale_lo, b.scale_hi);
  log_nu_m2 = Clamp(log_nu_m2, b.shape_lo, b.shape_hi);
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
  constexpr TailBounds b = tail_bounds(Distribution::SkewNormal);
  log_sigma2 = Clamp(log_sigma2, b.scale_lo, b.scale_hi);
  alpha = Clamp(alpha, b.shape_lo, b.shape_hi);
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
  constexpr TailBounds b = tail_bounds(Distribution::GED);
  log_sigma = Clamp(log_sigma, b.scale_lo, b.scale_hi);
  log_beta = Clamp(log_beta, b.shape_lo, b.shape_hi);
  double sigma = std::exp(log_sigma);
  double beta_ged = std::exp(log_beta);
  double sum_pow = 0.0;
  for (int i = 0; i < n; ++i)
    sum_pow += std::pow(std::abs(e[i]) / sigma, beta_ged);
  return (std::log(2.0) + log_sigma + std::lgamma(1.0 / beta_ged) - log_beta +
          sum_pow / static_cast<double>(n));
}

} // namespace dist
