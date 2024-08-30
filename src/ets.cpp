#include <pybind11/pybind11.h>

#include "nelder_mead.h"

namespace ets {
namespace py = pybind11;
using Eigen::VectorXd;

enum class Component {
  Nothing = 0,
  Additive = 1,
  Multiplicative = 2,
};
enum class Criterion {
  Likelihood = 0,
  MSE = 1,
  AMSE = 2,
  Sigma = 3,
  MAE = 4,
};
constexpr double HUGE_N = 1e10;
constexpr double NA = -99999.0;
constexpr double TOL = 1e-10;

template <typename Ref, typename CRef>
std::tuple<double, double>
Update(Ref s, double l, double b, double old_l, double old_b, CRef old_s, int m,
       Component trend, Component season, double alpha, double beta,
       double gamma, double phi, double y) {
  double q, phi_b;
  // new level
  if (trend == Component::Nothing) {
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
  if (season == Component::Nothing) {
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
  if (trend != Component::Nothing) {
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
  if (season != Component::Nothing) {
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
    std::copy(old_s.data(), old_s.data() + m - 1, s.data() + 1);
  }
  return {l, b};
}

template <typename Ref, typename CRef>
void Forecast(Ref f, double l, double b, CRef s, int m, Component trend,
              Component season, double phi, int h) {
  double phistar = phi;
  for (int i = 0; i < h; ++i) {
    if (trend == Component::Nothing) {
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

template <typename Ref, typename CRef>
double Calc(Ref x, Ref e, Ref a_mse, int n_mse, CRef y, Component error,
            Component trend, Component season, double alpha, double beta,
            double gamma, double phi, int m) {
  auto n = y.size();
  int n_s = std::max(m, 24);
  m = std::max(m, 1);
  n_mse = std::min(n_mse, 30);
  int n_states =
      m * (season != Component::Nothing) + (trend != Component::Nothing) + 1;

  // copy initial state components
  double l = x[0];
  double b = (trend != Component::Nothing) ? x[1] : 0.0;
  auto s = std::vector<double>(n_s);
  if (season != Component::Nothing) {
    std::copy(x.data() + 1 + (trend != Component::Nothing),
              x.data() + 1 + (trend != Component::Nothing) + m, s.data());
  }

  std::fill(a_mse.data(), a_mse.data() + n_mse, 0.0);
  auto old_s = std::vector<double>(n_s);
  auto denom = std::vector<double>(30);
  auto f = std::vector<double>(30);
  double old_b = 0.0;
  double lik = 0.0;
  double lik2 = 0.0;
  double f_0, old_l, val;
  for (Eigen::Index i = 0; i < n; ++i) {
    // copy previous state
    old_l = l;
    if (trend != Component::Nothing) {
      old_b = b;
    }
    if (season != Component::Nothing) {
      std::copy(s.begin(), s.begin() + m, old_s.begin());
    }
    // one step forecast
    Forecast<std::vector<double> &, const std::vector<double> &>(
        f, old_l, old_b, old_s, m, trend, season, phi, n_mse);
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
    for (Eigen::Index j = 0; j < n_mse; ++j) {
      if (i + j < y.size()) {
        denom[j] += 1.0;
        double tmp = y[i + j] - f[j];
        a_mse[j] = (a_mse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    // update state
    std::tie(l, b) = Update<std::vector<double> &, const std::vector<double> &>(
        s, l, b, old_l, old_b, old_s, m, trend, season, alpha, beta, gamma, phi,
        y[i]);

    // store new state
    x[n_states * (i + 1)] = l;
    if (trend != Component::Nothing) {
      x[n_states * (i + 1) + 1] = b;
    }
    if (season != Component::Nothing) {
      std::copy(s.data(), s.data() + m,
                x.data() + n_states * (i + 1) + 1 +
                    (trend != Component::Nothing));
    }
    lik += e[i] * e[i];
    val = std::abs(f[0]);
    if (val > 0.0) {
      lik2 += std::log(val);
    } else {
      lik2 += std::log(val + 1e-8);
    }
  }
  if (lik > 0.0) {
    lik = static_cast<double>(n) * std::log(lik);
  } else {
    lik = static_cast<double>(n) * std::log(lik + 1e-8);
  }
  if (error == Component::Multiplicative) {
    lik += 2 * lik2;
  }
  return lik;
}

double ObjectiveFunction(const VectorXd &params, const VectorXd &y, int n_state,
                         Component error, Component trend, Component season,
                         Criterion opt_crit, int n_mse, int m, bool opt_alpha,
                         bool opt_beta, bool opt_gamma, bool opt_phi,
                         double alpha, double beta, double gamma, double phi) {
  int j = 0;
  if (opt_alpha) {
    alpha = params(j++);
  }
  if (opt_beta) {
    beta = params(j++);
  }
  if (opt_gamma) {
    gamma = params(j++);
  }
  if (opt_phi) {
    phi = params(j++);
  }
  auto n_params = params.size();
  auto n = y.size();
  int p = n_state + (season != Component::Nothing);
  VectorXd state = VectorXd::Zero(p * (n + 1));
  std::copy(params.data() + n_params - n_state, params.data() + n_params,
            state.data());
  if (season != Component::Nothing) {
    // add extra state
    int start = 1 + (trend != Component::Nothing);
    double sum = state(Eigen::seq(start, n_state - 1)).sum();
    state(n_state) =
        static_cast<double>(m * (season == Component::Multiplicative)) - sum;
    if (season == Component::Multiplicative &&
        state.tail(state.size() - start).minCoeff() < 0.0) {
      return std::numeric_limits<double>::infinity();
    }
  }
  VectorXd a_mse = VectorXd::Zero(30);
  VectorXd e = VectorXd::Zero(n);
  double lik = Calc<VectorXd &, const VectorXd &>(state, e, a_mse, n_mse, y,
                                                  error, trend, season, alpha,
                                                  beta, gamma, phi, m);
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
    obj_val = a_mse(0);
    break;
  case Criterion::AMSE:
    obj_val = a_mse.head(n_mse).mean();
    break;
  case Criterion::Sigma:
    obj_val = e.array().square().mean();
    break;
  case Criterion::MAE:
    obj_val = e.array().abs().mean();
    break;
  }
  return obj_val;
}
std::tuple<VectorXd, double, int>
Optimize(const Eigen::Ref<const VectorXd> &x0,
         const Eigen::Ref<const VectorXd> &y, int n_state, Component error,
         Component trend, Component season, Criterion opt_crit, int n_mse,
         int m, bool opt_alpha, bool opt_beta, bool opt_gamma, bool opt_phi,
         double alpha, double beta, double gamma, double phi,
         const Eigen::Ref<const VectorXd> &lower,
         const Eigen::Ref<const VectorXd> &upper, double tol_std, int max_iter,
         bool adaptive) {
  double init_step = 0.05;
  double nm_alpha = 1.0;
  double nm_gamma = 2.0;
  double nm_rho = 0.5;
  double nm_sigma = 0.5;
  double zero_pert = 1.0e-4;
  return nm::NelderMead(ObjectiveFunction, x0, lower, upper, init_step,
                        zero_pert, nm_alpha, nm_gamma, nm_rho, nm_sigma,
                        max_iter, tol_std, adaptive, y, n_state, error, trend,
                        season, opt_crit, n_mse, m, opt_alpha, opt_beta,
                        opt_gamma, opt_phi, alpha, beta, gamma, phi);
}

void init(py::module_ &m) {
  py::module_ ets = m.def_submodule("ets");
  ets.attr("HUGE_N") = HUGE_N;
  ets.attr("NA") = NA;
  ets.attr("TOL") = TOL;
  py::enum_<Component>(ets, "Component")
      .value("Nothing", Component::Nothing)
      .value("Additive", Component::Additive)
      .value("Multiplicative", Component::Multiplicative);
  py::enum_<Criterion>(ets, "Criterion")
      .value("Likelihood", Criterion::Likelihood)
      .value("MSE", Criterion::MSE)
      .value("AMSE", Criterion::AMSE)
      .value("Sigma", Criterion::Sigma)
      .value("MAE", Criterion::MAE);
  ets.def("update",
          &Update<Eigen::Ref<VectorXd>, const Eigen::Ref<const VectorXd> &>);
  ets.def("forecast",
          &Forecast<Eigen::Ref<VectorXd>, const Eigen::Ref<const VectorXd> &>);
  ets.def("calc",
          &Calc<Eigen::Ref<VectorXd>, const Eigen::Ref<const VectorXd> &>);
  ets.def("optimize", &Optimize);
}
} // namespace ets
