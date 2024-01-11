#pragma once

#include <cmath>
#include <string>

#include "nelder_mead.h"

enum class Component {
  None = 0,
  Additive = 1,
  Multiplicative = 2,
};
enum class Criterion {
  Likelihood = 0,
  MSE = 1,
  AMSE = 2,
  SIGMA = 3,
  MAE = 4,
};
const double HUGE_N = 1e10;
const double NA = -99999.0;
const double TOL = 1e-10;

extern "C" OptimResult
ETS_NelderMead(double *x0, size_t n_x0, const double *y, size_t n_y,
               int n_state, Component error, Component trend, Component season,
               Criterion opt_crit, int n_mse, int m, bool opt_alpha,
               bool opt_beta, bool opt_gamma, bool opt_phi, double alpha,
               double beta, double gamma, double phi, const double *lower,
               const double *upper, double tol_std, int max_iter,
               bool adaptive);

extern "C" double
ETS_TargetFunction(double *params, size_t n_params, const double *y, size_t n_y,
                   int n_state, Component error, Component trend,
                   Component season, Criterion opt_crit, int n_mse, int m,
                   bool opt_alpha, bool opt_beta, bool opt_gamma, bool opt_phi,
                   double alpha, double beta, double gamma, double phi);
