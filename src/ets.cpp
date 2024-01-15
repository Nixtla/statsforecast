#include <algorithm>
#include <numeric>

#include "ets.h"
#include "nelder_mead.h"

namespace ETS {
double TargetFunction(const double *params, size_t n_params, const double *y,
                      size_t n, int n_state, Component error, Component trend,
                      Component season, Criterion opt_crit, int n_mse, int m,
                      bool opt_alpha, bool opt_beta, bool opt_gamma,
                      bool opt_phi, double alpha, double beta, double gamma,
                      double phi) {
  int j = 0;
  if (opt_alpha) {
    alpha = params[j];
    j++;
  }
  if (opt_beta) {
    beta = params[j];
    j++;
  }
  if (opt_gamma) {
    gamma = params[j];
    j++;
  }
  if (opt_phi) {
    phi = params[j];
    j++;
  }
  int p = n_state + (season != Component::None);
  auto state = std::vector<double>(p * (n + 1), 0.0);
  std::copy(params + n_params - n_state, params + n_params, state.begin());
  if (season != Component::None) {
    // add extra state
    int start = 1 + (trend != Component::None);
    double sum =
        std::accumulate(state.begin() + start, state.begin() + n_state, 0.0);
    state[n_state] =
        static_cast<double>(m * (season == Component::Multiplicative)) - sum;
    if (season == Component::Multiplicative &&
        *std::min_element(state.begin(), state.begin() + start) < 0.0) {
      return std::numeric_limits<double>::infinity();
    }
  }
  auto a_mse = std::vector<double>(30, 0.0);
  auto e = std::vector<double>(n, 0.0);
  double lik = ETS_Calc(y, n, state.data(), error, trend, season, alpha, beta,
                        gamma, phi, e.data(), a_mse.data(), n_mse, m);
  lik = std::max(lik, -1e10);
  if (std::isnan(lik) || std::abs(lik + 99999.0) < 1e-7) {
    lik = -std::numeric_limits<double>::infinity();
  }
  double obj_val = 0.0;
  switch (opt_crit) {
  case Criterion::Likelihood:
    obj_val = lik;
    break;
  case Criterion::MSE:
    obj_val = a_mse[0];
    break;
  case Criterion::AMSE:
    obj_val =
        std::accumulate(a_mse.begin(), a_mse.begin() + n_mse, 0.0) / n_mse;
    break;
  case Criterion::SIGMA:
    obj_val = std::accumulate(
                  e.begin(), e.end(), 0.0,
                  [](double sum, double val) { return sum + val * val; }) /
              e.size();
    break;
  case Criterion::MAE:
    obj_val = std::accumulate(
                  e.begin(), e.end(), 0.0,
                  [](double sum, double val) { return sum + std::abs(val); }) /
              e.size();
    break;
  }
  return obj_val;
}
} // namespace ETS

void ETS_Update(double &l, double &b, double *s, double old_l, double old_b,
                const double *old_s, int m, Component trend, Component season,
                double alpha, double beta, double gamma, double phi, double y) {
  double q, phi_b;
  // new level
  if (trend == Component::None) {
    q = old_l;
    phi_b = 0.0;
  } else if (trend == Component::Additive) {
    phi_b = phi * old_b;
    q = old_l + phi_b;
  } else if (std::abs(phi - 1.0) < TOL) {
    phi_b = old_b;
    q = old_l * old_b;
  } else {
    phi_b = std::pow(old_b, phi);
    q = old_l * phi_b;
  }
  // season
  double p;
  if (season == Component::None) {
    p = y;
  } else if (season == Component::Additive) {
    p = y - old_s[m - 1];
  } else {
    if (std::abs(old_s[m - 1]) < TOL) {
      p = HUGE_N;
    } else {
      p = y / old_s[m - 1];
    }
  }
  l = q + alpha * (p - q);
  // new growth
  double r;
  if (trend > Component::None) {
    if (trend == Component::Additive) {
      r = l - old_l;
    } else {
      if (std::abs(old_l) < TOL) {
        r = HUGE_N;
      } else {
        r = l / old_l;
      }
    }
    b = phi_b + (beta / alpha) * (r - phi_b);
  }
  // new seasonal
  double t;
  if (season > Component::None) {
    if (season == Component::Additive) {
      t = y - q;
    } else {
      if (std::abs(q) < TOL) {
        t = HUGE_N;
      } else {
        t = y / q;
      }
    }
    s[0] = old_s[m - 1] + gamma * (t - old_s[m - 1]);
    std::copy(old_s, old_s + m - 1, s + 1);
  }
}

void ETS_Forecast(double l, double b, const double *s, int m, Component trend,
                  Component season, double phi, double *f, int h) {
  double phistar = phi;
  for (int i = 0; i < h; ++i) {
    if (trend == Component::None) {
      f[i] = l;
    } else if (trend == Component::Additive) {
      f[i] = l + phistar * b;
    } else if (b < 0) {
      f[i] = std::numeric_limits<double>::quiet_NaN();
    } else {
      f[i] = l * std::pow(b, phistar);
    }
    int j = m - 1 - i;
    while (j < 0) {
      j += m;
    }
    if (season == Component::Additive) {
      f[i] += s[j];
    } else if (season == Component::Multiplicative) {
      f[i] *= s[j];
    }
    if (i < h - 1) {
      if (std::abs(phi - 1.0) < TOL) {
        phistar += 1.0;
      } else {
        phistar += std::pow(phi, i + 1);
      }
    }
  }
}

double ETS_Calc(const double *y, size_t n, double *x, Component error,
                Component trend, Component season, double alpha, double beta,
                double gamma, double phi, double *e, double *a_mse, int n_mse,
                int m) {
  double old_b = 0.0;
  int n_s = std::max(m, 24);
  auto old_s = std::vector<double>(n_s, 0.0);
  auto s = std::vector<double>(n_s, 0.0);
  auto f = std::vector<double>(30, 0.0);
  auto denom = std::vector<double>(30, 0.0);
  m = std::max(m, 1);
  n_mse = std::min(n_mse, 30);
  int n_states = m * (season > Component::None) + (trend > Component::None) + 1;
  // copy initial state components
  double l = x[0];
  double b = (trend > Component::None) ? x[1] : 0.0;
  if (season > Component::None) {
    std::copy(x + 1 + (trend > Component::None),
              x + 1 + (trend > Component::None) + m, s.begin());
  }
  double lik = 0.0;
  double lik2 = 0.0;
  std::fill(a_mse, a_mse + n_mse, 0.0);
  std::fill(denom.begin(), denom.begin() + n_mse, 0.0);
  double old_l, f_0;
  for (size_t i = 0; i < n; ++i) {
    // copy previous state
    old_l = l;
    if (trend > Component::None) {
      old_b = b;
    }
    if (season > Component::None) {
      std::copy(s.begin(), s.begin() + m, old_s.begin());
    }
    // one step forecast
    ETS_Forecast(old_l, old_b, old_s.data(), m, trend, season, phi, f.data(),
                 n_mse);
    if (std::abs(f[0] - NA) < TOL) {
      return NA;
    }
    if (error == Component::Additive) {
      e[i] = y[i] - f[0];
    } else {
      if (std::abs(f[0]) < TOL) {
        f_0 = f[0] + TOL;
      } else {
        f_0 = f[0];
      }
      e[i] = (y[i] - f[0]) / f_0;
    }
    for (int j = 0; j < n_mse; ++j) {
      if (i + j < n) {
        denom[j] += 1.0;
        double tmp = y[i + j] - f[j];
        a_mse[j] = (a_mse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    // update state
    ETS_Update(l, b, s.data(), old_l, old_b, old_s.data(), m, trend, season,
               alpha, beta, gamma, phi, y[i]);
    // store new state
    x[n_states * (i + 1)] = l;
    if (trend > Component::None) {
      x[n_states * (i + 1) + 1] = b;
    }
    if (season > Component::None) {
      std::copy(s.begin(), s.begin() + m,
                x + n_states * (i + 1) + 1 + (trend > Component::None));
    }
    lik += e[i] * e[i];
    double val = std::abs(f[0]);
    if (val > 0.0) {
      lik2 += std::log(val);
    } else {
      lik2 += std::log(val + 1e-8);
    }
  }
  if (lik > 0.0) {
    lik = n * std::log(lik);
  } else {
    lik = n * std::log(lik + 1e-8);
  }
  if (error == Component::Multiplicative) {
    lik += 2 * lik2;
  }
  return lik;
}

OptimResult ETS_NelderMead(double *x0, size_t n_x0, const double *y, size_t n_y,
                           int n_state, Component error, Component trend,
                           Component season, Criterion opt_crit, int n_mse,
                           int m, bool opt_alpha, bool opt_beta, bool opt_gamma,
                           bool opt_phi, double alpha, double beta,
                           double gamma, double phi, const double *lower,
                           const double *upper, double tol_std, int max_iter,
                           bool adaptive) {
  double init_step = 0.05;
  double nm_alpha = 1.0;
  double nm_gamma = 2.0;
  double nm_rho = 0.5;
  double nm_sigma = 0.5;
  double zero_pert = 1.0e-4;
  return NelderMead(ETS::TargetFunction, x0, n_x0, lower, upper, init_step,
                    zero_pert, nm_alpha, nm_gamma, nm_rho, nm_sigma, max_iter,
                    tol_std, adaptive, y, n_y, n_state, error, trend, season,
                    opt_crit, n_mse, m, opt_alpha, opt_beta, opt_gamma, opt_phi,
                    alpha, beta, gamma, phi);
}
