#pragma once

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
constexpr double HUGE_N = 1e10;
constexpr double NA = -99999.0;
constexpr double TOL = 1e-10;

extern "C" {
void ETS_Update(double &l, double &b, double *s, double old_l, double old_b,
                const double *old_s, int m, Component trend, Component season,
                double alpha, double beta, double gamma, double phi, double y);

void ETS_Forecast(double l, double b, const double *s, int m, Component trend,
                  Component season, double phi, double *f, int h);

double ETS_Calc(const double *y, size_t n, double *x, Component error,
                Component trend, Component season, double alpha, double beta,
                double gamma, double phi, double *e, double *a_mse, int n_mse,
                int m);

OptimResult ETS_NelderMead(double *x0, size_t n_x0, const double *y, size_t n_y,
                           int n_state, Component error, Component trend,
                           Component season, Criterion opt_crit, int n_mse,
                           int m, bool opt_alpha, bool opt_beta, bool opt_gamma,
                           bool opt_phi, double alpha, double beta,
                           double gamma, double phi, const double *lower,
                           const double *upper, double tol_std, int max_iter,
                           bool adaptive);
}
