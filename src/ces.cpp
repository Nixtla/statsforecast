#include <algorithm>
#include <cmath>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "nelder_mead.h"

namespace ces {
namespace py = pybind11;
using Eigen::VectorXd;
using RowMajorMatrixXf =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Season type constants
constexpr int NONE = 0;
constexpr int SIMPLE = 1;
constexpr int PARTIAL = 2;
constexpr int FULL = 3;
constexpr double TOL = 1.0e-10;
constexpr double NA = -99999.0;

int switch_ces(const std::string &x) {
  if (x == "N") return NONE;
  if (x == "S") return SIMPLE;
  if (x == "P") return PARTIAL;
  if (x == "F") return FULL;
  throw std::invalid_argument("Unknown season type: " + x);
}

void cesupdate(Eigen::Ref<RowMajorMatrixXf> states, Eigen::Index i, int m,
               int season, double alpha_0, double alpha_1, double beta_0,
               double beta_1, float y) {
  float e;
  if (season == NONE || season == PARTIAL || season == FULL) {
    e = y - states(i - 1, 0);
  } else {
    e = y - states(i - m, 0);
  }
  if (season > SIMPLE) {
    e -= states(i - m, 2);
  }

  if (season == NONE || season == PARTIAL || season == FULL) {
    states(i, 0) = states(i - 1, 0) - (1.0f - (float)alpha_1) * states(i - 1, 1) +
                   (float)(alpha_0 - alpha_1) * e;
    states(i, 1) = states(i - 1, 0) + (1.0f - (float)alpha_0) * states(i - 1, 1) +
                   (float)(alpha_0 + alpha_1) * e;
  } else {
    states(i, 0) = states(i - m, 0) - (1.0f - (float)alpha_1) * states(i - m, 1) +
                   (float)(alpha_0 - alpha_1) * e;
    states(i, 1) = states(i - m, 0) + (1.0f - (float)alpha_0) * states(i - m, 1) +
                   (float)(alpha_0 + alpha_1) * e;
  }

  if (season == PARTIAL) {
    states(i, 2) = states(i - m, 2) + (float)beta_0 * e;
  }
  if (season == FULL) {
    states(i, 2) = states(i - m, 2) - (1.0f - (float)beta_1) * states(i - m, 3) +
                   (float)(beta_0 - beta_1) * e;
    states(i, 3) = states(i - m, 2) + (1.0f - (float)beta_0) * states(i - m, 3) +
                   (float)(beta_0 + beta_1) * e;
  }
}

// Returns new_states matrix
RowMajorMatrixXf cesfcst(const Eigen::Ref<const RowMajorMatrixXf> &states,
                         Eigen::Index i, int m, int season,
                         Eigen::Ref<VectorXd> f, Eigen::Index h,
                         double alpha_0, double alpha_1, double beta_0,
                         double beta_1) {
  Eigen::Index cols = states.cols();
  RowMajorMatrixXf new_states = RowMajorMatrixXf::Zero(m + h, cols);
  new_states.topRows(m) = states.middleRows(i - m, m);

  for (Eigen::Index i_h = m; i_h < m + h; ++i_h) {
    if (season == NONE || season == PARTIAL || season == FULL) {
      f[i_h - m] = new_states(i_h - 1, 0);
    } else {
      f[i_h - m] = new_states(i_h - m, 0);
    }
    if (season > SIMPLE) {
      f[i_h - m] += new_states(i_h - m, 2);
    }
    cesupdate(new_states, i_h, m, season, alpha_0, alpha_1, beta_0, beta_1,
              (float)f[i_h - m]);
  }
  return new_states;
}

double cescalc(const Eigen::Ref<const VectorXd> &y,
               Eigen::Ref<RowMajorMatrixXf> states, int m, int season,
               double alpha_0, double alpha_1, double beta_0, double beta_1,
               Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, int nmse,
               int backfit) {
  VectorXd denom = VectorXd::Zero(nmse);
  int m_eff = (season == NONE) ? 1 : m;
  VectorXd f = VectorXd::Zero(std::max(nmse, m_eff));
  double lik = 0.0;
  amse.head(nmse).setZero();
  Eigen::Index n = y.size();

  // Create mutable copy of y for potential reversal
  VectorXd y_mut = y;

  for (Eigen::Index i = m_eff; i < n + m_eff; ++i) {
    auto new_states =
        cesfcst(states, i, m_eff, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1);
    if (std::fabs(f[0] - NA) < TOL) {
      return NA;
    }
    e[i - m_eff] = y_mut[i - m_eff] - f[0];
    for (int j = 0; j < nmse; ++j) {
      if ((i + j) < n) {
        denom[j] += 1.0;
        double tmp = y_mut[i + j] - f[j];
        amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    cesupdate(states, i, m_eff, season, alpha_0, alpha_1, beta_0, beta_1,
              (float)y_mut[i - m_eff]);
    lik += e[i - m_eff] * e[i - m_eff];
  }
  // Update last states
  auto ns = cesfcst(states, n + m_eff, m_eff, season, f, m_eff, alpha_0, alpha_1,
                    beta_0, beta_1);
  states.bottomRows(m_eff) = ns.bottomRows(m_eff);
  lik = n * std::log(lik);

  if (!backfit) {
    return lik;
  }

  // Backfit: reverse y, states, e
  y_mut.reverseInPlace();
  // Reverse the states rows in-place
  RowMajorMatrixXf states_reversed = states.colwise().reverse().eval();
  states = states_reversed;
  e.reverseInPlace();

  lik = 0.0;
  for (Eigen::Index i = m_eff; i < n + m_eff; ++i) {
    cesfcst(states, i, m_eff, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1);
    if (std::fabs(f[0] - NA) < TOL) {
      return NA;
    }
    e[i - m_eff] = y_mut[i - m_eff] - f[0];
    for (int j = 0; j < nmse; ++j) {
      if ((i + j) < n) {
        denom[j] += 1.0;
        double tmp = y_mut[i + j] - f[j];
        amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    cesupdate(states, i, m_eff, season, alpha_0, alpha_1, beta_0, beta_1,
              (float)y_mut[i - m_eff]);
    lik += e[i - m_eff] * e[i - m_eff];
  }
  ns = cesfcst(states, n + m_eff, m_eff, season, f, m_eff, alpha_0, alpha_1,
               beta_0, beta_1);
  states.bottomRows(m_eff) = ns.bottomRows(m_eff);

  // Forward again
  y_mut.reverseInPlace();
  states_reversed = states.colwise().reverse().eval();
  states = states_reversed;
  e.reverseInPlace();

  lik = 0.0;
  for (Eigen::Index i = m_eff; i < n + m_eff; ++i) {
    cesfcst(states, i, m_eff, season, f, nmse, alpha_0, alpha_1, beta_0, beta_1);
    if (std::fabs(f[0] - NA) < TOL) {
      return NA;
    }
    e[i - m_eff] = y_mut[i - m_eff] - f[0];
    for (int j = 0; j < nmse; ++j) {
      if ((i + j) < n) {
        denom[j] += 1.0;
        double tmp = y_mut[i + j] - f[j];
        amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    cesupdate(states, i, m_eff, season, alpha_0, alpha_1, beta_0, beta_1,
              (float)y_mut[i - m_eff]);
    lik += e[i - m_eff] * e[i - m_eff];
  }
  ns = cesfcst(states, n + m_eff, m_eff, season, f, m_eff, alpha_0, alpha_1,
               beta_0, beta_1);
  states.bottomRows(m_eff) = ns.bottomRows(m_eff);
  lik = n * std::log(lik);
  return lik;
}

// Target function for optimization
double ces_target_fn(const VectorXd &optimal_param, double init_alpha_0,
                     double init_alpha_1, double init_beta_0,
                     double init_beta_1, bool opt_alpha_0, bool opt_alpha_1,
                     bool opt_beta_0, bool opt_beta_1, const VectorXd &y,
                     int m, const RowMajorMatrixXf &init_states,
                     int n_components, int season, int nmse) {
  Eigen::Index n = y.size();
  RowMajorMatrixXf states = RowMajorMatrixXf::Zero(n + 2 * m, n_components);
  states.topRows(m) = init_states;

  Eigen::Index j = 0;
  double alpha_0 = opt_alpha_0 ? optimal_param[j++] : init_alpha_0;
  double alpha_1 = opt_alpha_1 ? optimal_param[j++] : init_alpha_1;
  double beta_0 = opt_beta_0 ? optimal_param[j++] : init_beta_0;
  double beta_1 = opt_beta_1 ? optimal_param[j++] : init_beta_1;

  VectorXd e = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  VectorXd amse = VectorXd::Constant(nmse, std::numeric_limits<double>::quiet_NaN());

  double lik =
      cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, e, amse,
              nmse, 1);

  if (lik < -1e10)
    lik = -1e10;
  if (std::isnan(lik))
    lik = -std::numeric_limits<double>::infinity();
  if (std::fabs(lik + 99999) < 1e-7)
    lik = -std::numeric_limits<double>::infinity();
  return lik;
}

// Optimization function exposed to Python
nm::OptimResult
optimize(const Eigen::Ref<const VectorXd> &x0,
         const Eigen::Ref<const VectorXd> &lower,
         const Eigen::Ref<const VectorXd> &upper, double init_alpha_0,
         double init_alpha_1, double init_beta_0, double init_beta_1,
         bool opt_alpha_0, bool opt_alpha_1, bool opt_beta_0, bool opt_beta_1,
         const Eigen::Ref<const VectorXd> &y, int m,
         const Eigen::Ref<const RowMajorMatrixXf> &init_states,
         int n_components, const std::string &seasontype, int nmse) {
  int season = switch_ces(seasontype);
  // Make a copy of init_states since target_fn needs to copy it each iteration
  RowMajorMatrixXf init_states_copy = init_states;
  return nm::NelderMead(
      ces_target_fn, x0, lower, upper, 0.05, 1e-4, 1.0, 2.0, 0.5, 0.5, 1000,
      1e-4, true, init_alpha_0, init_alpha_1, init_beta_0, init_beta_1,
      opt_alpha_0, opt_alpha_1, opt_beta_0, opt_beta_1, y, m,
      init_states_copy, n_components, season, nmse);
}

// Pegels residuals exposed to Python
std::tuple<VectorXd, VectorXd, RowMajorMatrixXf, double>
pegelsresid(const Eigen::Ref<const VectorXd> &y, int m,
            const Eigen::Ref<const RowMajorMatrixXf> &init_states,
            int n_components, const std::string &seasontype, double alpha_0,
            double alpha_1, double beta_0, double beta_1, int nmse) {
  Eigen::Index n = y.size();
  RowMajorMatrixXf states =
      RowMajorMatrixXf::Zero(n + 2 * m, n_components);
  states.topRows(m) = init_states;

  VectorXd e = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  VectorXd amse = VectorXd::Constant(nmse, std::numeric_limits<double>::quiet_NaN());

  int season = switch_ces(seasontype);
  double lik =
      cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, e, amse,
              nmse, 1);

  if (!std::isnan(lik)) {
    if (std::fabs(lik + 99999) < 1e-7) {
      lik = std::numeric_limits<double>::quiet_NaN();
    }
  }
  return {amse, e, states, lik};
}

// Forecast function exposed to Python
RowMajorMatrixXf forecast(const Eigen::Ref<const RowMajorMatrixXf> &states,
                          Eigen::Index n, int m, const std::string &seasontype,
                          Eigen::Ref<VectorXd> f, Eigen::Index h,
                          double alpha_0, double alpha_1, double beta_0,
                          double beta_1) {
  int season = switch_ces(seasontype);
  int m_eff = (season == NONE) ? 1 : m;
  // Make mutable copy of states
  RowMajorMatrixXf states_copy = states;
  return cesfcst(states_copy, m_eff + n, m_eff, season, f, h, alpha_0,
                 alpha_1, beta_0, beta_1);
}

// cescalc exposed to Python (operates on pre-allocated arrays in place)
double cescalc_py(const Eigen::Ref<const VectorXd> &y,
                  Eigen::Ref<RowMajorMatrixXf> states, int m, int season,
                  double alpha_0, double alpha_1, double beta_0, double beta_1,
                  Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, int nmse,
                  int backfit) {
  return cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, e,
                 amse, nmse, backfit);
}

void init(py::module_ &m) {
  py::module_ ces_mod = m.def_submodule("ces");
  ces_mod.def("switch_ces", &switch_ces);
  ces_mod.def("optimize", &optimize);
  ces_mod.def("pegelsresid", &pegelsresid);
  ces_mod.def("forecast", &forecast);
  ces_mod.def("cescalc", &cescalc_py);
}

} // namespace ces
