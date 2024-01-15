#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct OptimResult {
  double fun;
  int nit;
};

inline void Clamp(double *x, size_t n_x, const double *lower,
                  const double *upper) {
  for (size_t i = 0; i < n_x; ++i) {
    x[i] = std::clamp(x[i], lower[i], upper[i]);
  }
}

inline std::vector<size_t> ArgSort(const std::vector<double> &v) {
  std::vector<size_t> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });
  return indices;
}

inline double StandardDeviation(const std::vector<double> &v) {
  size_t n = v.size();
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / n;
  double var = std::accumulate(v.begin(), v.end(), 0.0,
                               [mean](double sum, double val) {
                                 return sum + (val - mean) * (val - mean);
                               }) /
               n;
  return std::sqrt(var);
}

template <typename Func, typename... Args>
OptimResult NelderMead(Func F, double *x0, size_t n, const double *lower,
                       const double *upper, double init_step, double zero_pert,
                       double alpha, double gamma, double rho, double sigma,
                       int max_iter, double tol_std, bool adaptive,
                       Args &&...args) {
  Clamp(x0, n, lower, upper);
  if (adaptive) {
    gamma = 1.0 + 2.0 / n;
    rho = 0.75 - 1.0 / (2 * n);
    sigma = 1.0 - 1.0 / n;
  }
  auto simplex = std::vector<double>((n + 1) * n);
  for (size_t i = 0; i < n + 1; ++i) {
    std::copy(x0, x0 + n, simplex.begin() + i * n);
  }
  // perturb simplex using init_step
  for (size_t i = 0; i < n; ++i) {
    if (simplex[i * n + i] == 0.0) {
      simplex[i * n + i] = zero_pert;
    } else {
      simplex[i * n + i] *= 1.0 + init_step;
    }
    simplex[i * n + i] = std::clamp(simplex[i * n + i], lower[i], upper[i]);
  }
  // array of the value of f
  auto f_simplex = std::vector<double>(n + 1);
  for (size_t i = 0; i < n + 1; ++i) {
    f_simplex[i] = F(simplex.data() + i * n, n, std::forward<Args>(args)...);
  }
  int i;
  size_t best_idx = 0;
  for (i = 0; i < max_iter; ++i) {
    // check whether method should stop
    if (StandardDeviation(f_simplex) < tol_std) {
      break;
    }

    // Step1: order of f_simplex
    std::vector<size_t> order_f = ArgSort(f_simplex);
    best_idx = order_f[0];
    size_t worst_idx = order_f[n];
    size_t second_worst_idx = order_f[n - 1];
    // calculate centroid as the col means removing the row with the max fval
    std::vector<double> x_o(n, 0.0);
    for (size_t j = 0; j < n; ++j) {
      for (size_t k = 0; k < n; ++k) {
        x_o[k] += simplex[order_f[j] * n + k] / n;
      }
    }

    // Step2: Reflection, Compute reflected point
    std::vector<double> x_r(n);
    for (size_t j = 0; j < n; ++j) {
      x_r[j] = x_o[j] + alpha * (x_o[j] - simplex[worst_idx * n + j]);
    }
    Clamp(x_r.data(), x_r.size(), lower, upper);
    double f_r = F(x_r.data(), x_r.size(), std::forward<Args>(args)...);
    if (f_simplex[best_idx] <= f_r && f_r < f_simplex[second_worst_idx]) {
      // accept reflection point
      std::copy(x_r.begin(), x_r.end(), simplex.begin() + worst_idx * n);
      f_simplex[worst_idx] = f_r;
      continue;
    }

    // Step3: Expansion, reflected point is the best point so far
    if (f_r < f_simplex[best_idx]) {
      std::vector<double> x_e(n);
      for (size_t j = 0; j < n; ++j) {
        x_e[j] = x_o[j] + gamma * (x_r[j] - x_o[j]);
      }
      Clamp(x_e.data(), x_e.size(), lower, upper);
      double f_e = F(x_e.data(), x_e.size(), std::forward<Args>(args)...);
      if (f_e < f_r) {
        // accept expansion point
        std::copy(x_e.begin(), x_e.end(), simplex.begin() + worst_idx * n);
        f_simplex[worst_idx] = f_e;
      } else {
        // accept reflection point
        std::copy(x_r.begin(), x_r.end(), simplex.begin() + worst_idx * n);
        f_simplex[worst_idx] = f_r;
      }
      continue;
    }

    // Step4: Contraction
    if (f_simplex[second_worst_idx] <= f_r && f_r < f_simplex[worst_idx]) {
      std::vector<double> x_oc(n);
      for (size_t j = 0; j < n; ++j) {
        x_oc[j] = x_o[j] + rho * (x_r[j] - x_o[j]);
      }
      Clamp(x_oc.data(), x_oc.size(), lower, upper);
      double f_oc = F(x_oc.data(), x_oc.size(), std::forward<Args>(args)...);
      if (f_oc <= f_r) {
        // accept contraction point
        std::copy(x_oc.begin(), x_oc.end(), simplex.begin() + worst_idx * n);
        f_simplex[worst_idx] = f_oc;
        continue;
      }
    } else {
      // Step5: Inside contraction
      std::vector<double> x_ic(n);
      for (size_t j = 0; j < n; ++j) {
        x_ic[j] = x_o[j] - rho * (x_r[j] - x_o[j]);
      }
      Clamp(x_ic.data(), x_ic.size(), lower, upper);
      double f_ic = F(x_ic.data(), x_ic.size(), std::forward<Args>(args)...);
      if (f_ic < f_simplex[worst_idx]) {
        // accept inside contraction point
        std::copy(x_ic.begin(), x_ic.end(), simplex.begin() + worst_idx * n);
        f_simplex[worst_idx] = f_ic;
        continue;
      }
    }

    // Step6: Shrink
    for (size_t j = 1; j < n + 1; ++j) {
      for (size_t k = 0; k < n; ++k) {
        simplex[j * n + k] =
            simplex[best_idx * n + k] +
            sigma * (simplex[j * n + k] - simplex[best_idx * n + k]);
        simplex[j * n + k] = std::clamp(simplex[j * n + k], lower[k], upper[k]);
      }
      f_simplex[j] = F(simplex.data() + j * n, n, std::forward<Args>(args)...);
    }
  }
  std::copy(simplex.begin() + best_idx * n,
            simplex.begin() + (best_idx + 1) * n, x0);
  return OptimResult({f_simplex[best_idx], i});
}
