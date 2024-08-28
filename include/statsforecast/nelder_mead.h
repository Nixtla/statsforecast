#pragma once

#include <numeric>
#include <pybind11/eigen.h>

namespace nm {
using Eigen::VectorXd;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

auto Clamp(const VectorXd &x, const VectorXd &lower, const VectorXd &upper) {
  return x.cwiseMax(lower).cwiseMin(upper);
}

double StandardDeviation(const VectorXd &x) {
  return std::sqrt((x.array() - x.mean()).square().mean());
}

Eigen::VectorX<Eigen::Index> ArgSort(const VectorXd &v) {
  Eigen::VectorX<Eigen::Index> indices(v.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&v](Eigen::Index i1, Eigen::Index i2) { return v(i1) < v(i2); });
  return indices;
}

template <typename Func, typename... Args>
std::tuple<VectorXd, double, int> NelderMead(Func F, const VectorXd &x, const VectorXd &lower,
                         const VectorXd upper, double init_step,
                         double zero_pert, double alpha, double gamma,
                         double rho, double sigma, int max_iter, double tol_std,
                         bool adaptive, Args &&...args) {
  auto x0 = Clamp(x, lower, upper);
  auto n = x0.size();
  if (adaptive) {
    gamma = 1.0 + 2.0 / n;
    rho = 0.75 - 1.0 / (2 * n);
    sigma = 1.0 - 1.0 / n;
  }
  RowMajorMatrixXd simplex = x0.transpose().replicate(n + 1, 1);
  // perturb simplex using init_step
  for (Eigen::Index i = 0; i < n; ++i) {
    double val = simplex(i, i);
    if (val == 0) {
      val = zero_pert;
    } else {
      val *= 1.0 + init_step;
    }
    simplex(i, i) = std::clamp(val, lower(i), upper(i));
  }
  // array of the value of f
  auto f_simplex = VectorXd(simplex.rows());
  for (Eigen::Index i = 0; i < simplex.rows(); ++i) {
    f_simplex(i) = F(simplex.row(i), std::forward<Args>(args)...);
  }
  int i;
  Eigen::Index best_idx = 0;
  for (i = 0; i < max_iter; ++i) {
    // check whether method should stop
    if (StandardDeviation(f_simplex) < tol_std) {
      break;
    }

    // Step1: order of f_simplex
    Eigen::VectorX<Eigen::Index> order_f = ArgSort(f_simplex);
    best_idx = order_f(0);
    auto worst_idx = order_f(n);
    auto second_worst_idx = order_f(n - 1);

    // calculate centroid as the col means removing the row with the max fval
    VectorXd x_o = (simplex.colwise().sum() - simplex.row(worst_idx)) / n;

    // Step2: Reflection, Compute reflected point
    VectorXd x_r = x_o + alpha * (x_o - simplex.row(worst_idx).transpose());
    x_r = Clamp(x_r, lower, upper);
    double f_r = F(x_r, std::forward<Args>(args)...);
    if (f_simplex(best_idx) <= f_r && f_r < f_simplex(second_worst_idx)) {
      // accept reflection point
      simplex(worst_idx, Eigen::all) = x_r;
      f_simplex(worst_idx) = f_r;
      continue;
    }

    // Step3: Expansion, reflected point is the best point so far
    if (f_r < f_simplex(best_idx)) {
      VectorXd x_e = x_o + gamma * (x_r - x_o);
      x_e = Clamp(x_e, lower, upper);
      double f_e = F(x_e, std::forward<Args>(args)...);
      if (f_e < f_r) {
        // accept expansion point
        simplex(worst_idx, Eigen::all) = x_e;
        f_simplex(worst_idx) = f_e;
      } else {
        // accept reflection point
        simplex(worst_idx, Eigen::all) = x_r;
        f_simplex(worst_idx) = f_r;
      }
      continue;
    }

    // Step4: Contraction
    if (f_simplex(second_worst_idx) <= f_r && f_r < f_simplex(worst_idx)) {
      VectorXd x_oc = x_o + rho * (x_r - x_o);
      x_oc = Clamp(x_oc, lower, upper);
      double f_oc = F(x_oc, std::forward<Args>(args)...);
      if (f_oc <= f_r) {
        // accept contraction point
        simplex(worst_idx, Eigen::all) = x_oc;
        f_simplex(worst_idx) = f_oc;
        continue;
      }
    } else {
      // Step5: Inside contraction
      VectorXd x_ic = x_o - rho * (x_r - x_o);
      x_ic = Clamp(x_ic, lower, upper);
      double f_ic = F(x_ic, std::forward<Args>(args)...);
      if (f_ic < f_simplex(worst_idx)) {
        // accept inside contraction point
        simplex(worst_idx, Eigen::all) = x_ic;
        f_simplex(worst_idx) = f_ic;
        continue;
      }
    }

    // Step6: Shrink
    for (Eigen::Index j = 1; j < simplex.rows(); ++j) {
      simplex.row(j) = simplex.row(best_idx) +
                       sigma * (simplex.row(j) - simplex.row(best_idx));
      simplex.row(j) = Clamp(simplex.row(j), lower, upper);
      f_simplex(j) = F(simplex.row(j), std::forward<Args>(args)...);
    }
  }
  return {simplex.row(best_idx), f_simplex(best_idx), i + 1};
}
} // namespace nm
