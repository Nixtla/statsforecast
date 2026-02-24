#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "nelder_mead.h"

namespace ces {
namespace py = pybind11;
using Eigen::VectorXd;
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

void cesupdate(Eigen::Ref<RowMajorMatrixXd> states, Eigen::Index i, int m,
               int season, double alpha_0, double alpha_1, double beta_0,
               double beta_1, double y) {
  double e;
  if (season == NONE || season == PARTIAL || season == FULL) {
    e = y - states(i - 1, 0);
  } else {
    e = y - states(i - m, 0);
  }
  if (season > SIMPLE) {
    e -= states(i - m, 2);
  }

  if (season == NONE || season == PARTIAL || season == FULL) {
    double s0 = states(i - 1, 0), s1 = states(i - 1, 1);
    states(i, 0) = s0 - (1.0 - alpha_1) * s1 + (alpha_0 - alpha_1) * e;
    states(i, 1) = s0 + (1.0 - alpha_0) * s1 + (alpha_0 + alpha_1) * e;
  } else {
    double s0 = states(i - m, 0), s1 = states(i - m, 1);
    states(i, 0) = s0 - (1.0 - alpha_1) * s1 + (alpha_0 - alpha_1) * e;
    states(i, 1) = s0 + (1.0 - alpha_0) * s1 + (alpha_0 + alpha_1) * e;
  }

  if (season == PARTIAL) {
    states(i, 2) = states(i - m, 2) + beta_0 * e;
  }
  if (season == FULL) {
    double s2 = states(i - m, 2), s3 = states(i - m, 3);
    states(i, 2) = s2 - (1.0 - beta_1) * s3 + (beta_0 - beta_1) * e;
    states(i, 3) = s2 + (1.0 - beta_0) * s3 + (beta_0 + beta_1) * e;
  }
}

// Forecast into a pre-allocated buffer. Writes into buf (must have >= m+h
// rows). Returns a reference to buf for convenience.
void cesfcst_buf(Eigen::Ref<RowMajorMatrixXd> buf,
                 const Eigen::Ref<const RowMajorMatrixXd> &states,
                 Eigen::Index i, int m, int season, Eigen::Ref<VectorXd> f,
                 Eigen::Index h, double alpha_0, double alpha_1,
                 double beta_0, double beta_1) {
  buf.topRows(m) = states.middleRows(i - m, m);
  // Zero only the rows we'll write to (m .. m+h-1)
  buf.middleRows(m, h).setZero();

  for (Eigen::Index i_h = m; i_h < m + h; ++i_h) {
    if (season == NONE || season == PARTIAL || season == FULL) {
      f[i_h - m] = buf(i_h - 1, 0);
    } else {
      f[i_h - m] = buf(i_h - m, 0);
    }
    if (season > SIMPLE) {
      f[i_h - m] += buf(i_h - m, 2);
    }
    cesupdate(buf, i_h, m, season, alpha_0, alpha_1, beta_0, beta_1,
              f[i_h - m]);
  }
}

// Allocating version (for the public forecast API where a buffer isn't reused)
RowMajorMatrixXd cesfcst(const Eigen::Ref<const RowMajorMatrixXd> &states,
                         Eigen::Index i, int m, int season,
                         Eigen::Ref<VectorXd> f, Eigen::Index h,
                         double alpha_0, double alpha_1, double beta_0,
                         double beta_1) {
  Eigen::Index cols = states.cols();
  RowMajorMatrixXd new_states = RowMajorMatrixXd::Zero(m + h, cols);
  cesfcst_buf(new_states, states, i, m, season, f, h, alpha_0, alpha_1, beta_0,
              beta_1);
  return new_states;
}

// Helper: reverse states rows in-place by swapping top/bottom rows
static void reverse_rows_inplace(Eigen::Ref<RowMajorMatrixXd> mat) {
  Eigen::Index nrows = mat.rows();
  for (Eigen::Index i = 0; i < nrows / 2; ++i) {
    mat.row(i).swap(mat.row(nrows - 1 - i));
  }
}

// Inline multi-step forecast using scalar state variables.
// During forecasting the "observation" equals the forecast value, so the
// error inside cesupdate is identically zero.  This reduces state propagation
// to a pure linear recurrence that needs no matrix buffer.
// Requires nmse <= m_eff for seasonal types (always true in practice).
static void inline_forecast(const Eigen::Ref<const RowMajorMatrixXd> &states,
                            Eigen::Index i, int m_eff, int season,
                            Eigen::Ref<VectorXd> f, int nmse, double a0,
                            double a1) {
  double ca1 = 1.0 - a1; // complement of alpha_1
  double ca0 = 1.0 - a0; // complement of alpha_0

  if (season == SIMPLE) {
    // Each step reads directly from original states (no recurrence needed)
    for (int j = 0; j < nmse; ++j) {
      f[j] = states(i - m_eff + j, 0);
    }
  } else if (season == NONE) {
    // 2-component scalar recurrence in double precision
    double s0 = states(i - 1, 0), s1 = states(i - 1, 1);
    f[0] = s0;
    for (int j = 1; j < nmse; ++j) {
      double s0_new = s0 - ca1 * s1;
      double s1_new = s0 + ca0 * s1;
      s0 = s0_new;
      s1 = s1_new;
      f[j] = s0;
    }
  } else {
    // PARTIAL or FULL: level follows NONE recurrence, seasonal read from
    // original states (valid when nmse <= m_eff, which is always true since
    // nmse <= 30 and m_eff >= 12 for seasonal models)
    double s0 = states(i - 1, 0), s1 = states(i - 1, 1);
    f[0] = s0 + states(i - m_eff, 2);
    for (int j = 1; j < nmse; ++j) {
      double s0_new = s0 - ca1 * s1;
      double s1_new = s0 + ca0 * s1;
      s0 = s0_new;
      s1 = s1_new;
      f[j] = s0 + states(i - m_eff + j, 2);
    }
  }
}

// Core inner loop used by cescalc for one pass (forward or backward).
// Updates states, e, amse, denom in place. Returns accumulated lik.
static double cescalc_pass(const VectorXd &y_mut,
                           Eigen::Ref<RowMajorMatrixXd> states, int m_eff,
                           int season, double alpha_0, double alpha_1,
                           double beta_0, double beta_1,
                           Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse,
                           VectorXd &denom, int nmse, VectorXd &f) {
  Eigen::Index n = y_mut.size();
  double lik = 0.0;

  // inline_forecast is only valid for seasonal types when nmse <= m_eff.
  // Pre-allocate a fallback buffer for the rare case where nmse > m_eff.
  bool use_inline = (season <= SIMPLE || nmse <= m_eff);
  RowMajorMatrixXd fcst_fallback;
  if (!use_inline) {
    fcst_fallback = RowMajorMatrixXd::Zero(m_eff + nmse, states.cols());
  }

  for (Eigen::Index i = m_eff; i < n + m_eff; ++i) {
    // Multi-step forecast via zero-error scalar recurrence (fast path)
    // or buffer-based forecast (fallback for nmse > m_eff with seasonal)
    if (use_inline) {
      inline_forecast(states, i, m_eff, season, f, nmse, alpha_0, alpha_1);
    } else {
      cesfcst_buf(fcst_fallback, states, i, m_eff, season, f, nmse, alpha_0,
                   alpha_1, beta_0, beta_1);
    }
    if (std::fabs(f[0] - NA) < TOL) {
      return NA;
    }
    e[i - m_eff] = y_mut[i - m_eff] - f[0];
    for (int j = 0; j < nmse; ++j) {
      if ((i - m_eff + j) < n) {
        denom[j] += 1.0;
        double tmp = y_mut[i - m_eff + j] - f[j];
        amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    // Update state with actual observation
    cesupdate(states, i, m_eff, season, alpha_0, alpha_1, beta_0, beta_1,
              y_mut[i - m_eff]);
    lik += e[i - m_eff] * e[i - m_eff];
  }
  return lik;
}

// Update trailing states after a pass
static void update_trailing_states(Eigen::Ref<RowMajorMatrixXd> states,
                                   Eigen::Index n, int m_eff, int season,
                                   VectorXd &f, double alpha_0, double alpha_1,
                                   double beta_0, double beta_1,
                                   RowMajorMatrixXd &fcst_buf) {
  cesfcst_buf(fcst_buf, states, n + m_eff, m_eff, season, f, m_eff, alpha_0,
              alpha_1, beta_0, beta_1);
  states.bottomRows(m_eff) = fcst_buf.middleRows(m_eff, m_eff);
}

double cescalc(const Eigen::Ref<const VectorXd> &y,
               Eigen::Ref<RowMajorMatrixXd> states, int m, int season,
               double alpha_0, double alpha_1, double beta_0, double beta_1,
               Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, int nmse,
               int backfit) {
  VectorXd denom = VectorXd::Zero(nmse);
  int m_eff = (season == NONE) ? 1 : m;
  VectorXd f = VectorXd::Zero(std::max(nmse, m_eff));
  amse.head(nmse).setZero();
  Eigen::Index n = y.size();
  Eigen::Index cols = states.cols();

  // Create mutable copy of y for potential reversal
  VectorXd y_mut = y;

  // Buffer only needed for update_trailing_states (not the inner loop)
  RowMajorMatrixXd fcst_buf = RowMajorMatrixXd::Zero(2 * m_eff, cols);

  // First forward pass
  double lik = cescalc_pass(y_mut, states, m_eff, season, alpha_0, alpha_1,
                            beta_0, beta_1, e, amse, denom, nmse, f);
  if (std::fabs(lik - NA) < TOL) {
    return NA;
  }
  update_trailing_states(states, n, m_eff, season, f, alpha_0, alpha_1, beta_0,
                         beta_1, fcst_buf);
  lik = n * std::log(lik);

  if (!backfit) {
    return lik;
  }

  // Backfit: reverse y, states, e
  y_mut.reverseInPlace();
  reverse_rows_inplace(states);
  e.reverseInPlace();

  lik = cescalc_pass(y_mut, states, m_eff, season, alpha_0, alpha_1, beta_0,
                     beta_1, e, amse, denom, nmse, f);
  if (std::fabs(lik - NA) < TOL) {
    return NA;
  }
  update_trailing_states(states, n, m_eff, season, f, alpha_0, alpha_1, beta_0,
                         beta_1, fcst_buf);

  // Forward again
  y_mut.reverseInPlace();
  reverse_rows_inplace(states);
  e.reverseInPlace();

  lik = cescalc_pass(y_mut, states, m_eff, season, alpha_0, alpha_1, beta_0,
                     beta_1, e, amse, denom, nmse, f);
  if (std::fabs(lik - NA) < TOL) {
    return NA;
  }
  update_trailing_states(states, n, m_eff, season, f, alpha_0, alpha_1, beta_0,
                         beta_1, fcst_buf);
  lik = n * std::log(lik);
  return lik;
}

// Target function for optimization
double ces_target_fn(const VectorXd &optimal_param, double init_alpha_0,
                     double init_alpha_1, double init_beta_0,
                     double init_beta_1, bool opt_alpha_0, bool opt_alpha_1,
                     bool opt_beta_0, bool opt_beta_1, const VectorXd &y,
                     int m, const RowMajorMatrixXd &init_states,
                     int n_components, int season, int nmse) {
  Eigen::Index n = y.size();
  RowMajorMatrixXd states = RowMajorMatrixXd::Zero(n + 2 * m, n_components);
  states.topRows(m) = init_states;

  Eigen::Index j = 0;
  double alpha_0 = opt_alpha_0 ? optimal_param[j++] : init_alpha_0;
  double alpha_1 = opt_alpha_1 ? optimal_param[j++] : init_alpha_1;
  double beta_0 = opt_beta_0 ? optimal_param[j++] : init_beta_0;
  double beta_1 = opt_beta_1 ? optimal_param[j++] : init_beta_1;

  VectorXd e = VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  VectorXd amse =
      VectorXd::Constant(nmse, std::numeric_limits<double>::quiet_NaN());

  double lik = cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1,
                       e, amse, nmse, 1);

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
         const Eigen::Ref<const RowMajorMatrixXd> &init_states,
         int n_components, const std::string &seasontype, int nmse) {
  int season = switch_ces(seasontype);
  // Make a copy of init_states since target_fn needs to copy it each iteration
  RowMajorMatrixXd init_states_copy = init_states;
  return nm::NelderMead(
      ces_target_fn, x0, lower, upper, 0.05, 1e-4, 1.0, 2.0, 0.5, 0.5, 1000,
      1e-4, true, init_alpha_0, init_alpha_1, init_beta_0, init_beta_1,
      opt_alpha_0, opt_alpha_1, opt_beta_0, opt_beta_1, y, m,
      init_states_copy, n_components, season, nmse);
}

// Pegels residuals exposed to Python
std::tuple<VectorXd, VectorXd, RowMajorMatrixXd, double>
pegelsresid(const Eigen::Ref<const VectorXd> &y, int m,
            const Eigen::Ref<const RowMajorMatrixXd> &init_states,
            int n_components, const std::string &seasontype, double alpha_0,
            double alpha_1, double beta_0, double beta_1, int nmse) {
  Eigen::Index n = y.size();
  RowMajorMatrixXd states = RowMajorMatrixXd::Zero(n + 2 * m, n_components);
  states.topRows(m) = init_states;

  VectorXd e =
      VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
  VectorXd amse =
      VectorXd::Constant(nmse, std::numeric_limits<double>::quiet_NaN());

  int season = switch_ces(seasontype);
  double lik = cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1,
                       e, amse, nmse, 1);

  if (!std::isnan(lik)) {
    if (std::fabs(lik + 99999) < 1e-7) {
      lik = std::numeric_limits<double>::quiet_NaN();
    }
  }
  return {amse, e, states, lik};
}

// Forecast function exposed to Python
RowMajorMatrixXd forecast(const Eigen::Ref<const RowMajorMatrixXd> &states,
                          Eigen::Index n, int m, const std::string &seasontype,
                          Eigen::Ref<VectorXd> f, Eigen::Index h,
                          double alpha_0, double alpha_1, double beta_0,
                          double beta_1) {
  int season = switch_ces(seasontype);
  int m_eff = (season == NONE) ? 1 : m;
  // Make mutable copy of states
  RowMajorMatrixXd states_copy = states;
  return cesfcst(states_copy, m_eff + n, m_eff, season, f, h, alpha_0,
                 alpha_1, beta_0, beta_1);
}

// cescalc exposed to Python (operates on pre-allocated arrays in place)
double cescalc_py(const Eigen::Ref<const VectorXd> &y,
                  Eigen::Ref<RowMajorMatrixXd> states, int m, int season,
                  double alpha_0, double alpha_1, double beta_0, double beta_1,
                  Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, int nmse,
                  int backfit) {
  return cescalc(y, states, m, season, alpha_0, alpha_1, beta_0, beta_1, e,
                 amse, nmse, backfit);
}

void init(py::module_ &m) {
  py::module_ ces_mod = m.def_submodule("ces");
  ces_mod.def("switch_ces", &switch_ces);
  ces_mod.def("optimize", &optimize,
              py::call_guard<py::gil_scoped_release>());
  ces_mod.def("pegelsresid", &pegelsresid,
              py::call_guard<py::gil_scoped_release>());
  ces_mod.def("forecast", &forecast,
              py::call_guard<py::gil_scoped_release>());
  ces_mod.def("cescalc", &cescalc_py,
              py::call_guard<py::gil_scoped_release>());
}

} // namespace ces
