#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>
#include <ranges>

#include "distributions.h"
#include "nelder_mead.h"

namespace theta {
namespace py = pybind11;
using Eigen::VectorXd;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum class ModelType { STM, OTM, DSTM, DOTM };
constexpr double HUGE_N = 1e10;
constexpr double NA = -99999.0;
constexpr double TOL = 1e-10;

Eigen::Vector<double, 5> init_state(const Eigen::Ref<const VectorXd> &y,
                                    ModelType model_type,
                                    double initial_smoothed, double alpha,
                                    double theta) {
  double An, Bn, mu;
  if (model_type == ModelType::DSTM || model_type == ModelType::DOTM) {
    An = y[0];
    Bn = double{};
    mu = y[0];
  } else {
    size_t n = y.size();
    double y_mean = y.array().mean();
    double weighted_avg = y.dot(VectorXd::LinSpaced(y.size(), 1, y.size())) / n;
    Bn = (6 * (2 * weighted_avg - (n + 1) * y_mean)) / (n * n - 1);
    An = y_mean - (n + 1) * Bn / 2;
    mu = initial_smoothed + (1 - 1 / theta) * (An + Bn);
  }
  return {alpha * y[0] + (1 - alpha) * initial_smoothed, y[0], An, Bn, mu};
}

// One state transition, reading the previous row and writing the next one.
// `i` enters the equations only as the time index.
void update_step(const double *prev, double *cur, size_t i,
                 ModelType model_type, double alpha, double theta, double y,
                 bool usemu) {
  double level = prev[0];
  double meany = prev[1];
  double An = prev[2];
  double Bn = prev[3];
  cur[4] =
      level + (1 - 1 / theta) * (An * std::pow(1 - alpha, i) +
                                 Bn * (1 - std::pow(1 - alpha, i + 1)) / alpha);
  if (usemu) {
    y = cur[4];
  }
  cur[0] = alpha * y + (1 - alpha) * level;
  cur[1] = (i * meany + y) / (i + 1);
  if (model_type == ModelType::DSTM || model_type == ModelType::DOTM) {
    cur[3] = ((i - 1) * Bn + 6 * (y - meany) / (i + 1)) / (i + 2);
    cur[2] = cur[1] - cur[3] * (i + 2) / 2;
  } else {
    cur[2] = An;
    cur[3] = Bn;
  }
}

void update(Eigen::Ref<RowMajorMatrixXd> states, size_t i, ModelType model_type,
            double alpha, double theta, double y, bool usemu) {
  update_step(states.row(i - 1).data(), states.row(i).data(), i, model_type,
              alpha, theta, y, usemu);
}

void forecast(const Eigen::Ref<const RowMajorMatrixXd> &states, size_t i,
              ModelType model_type, Eigen::Ref<VectorXd> f, double alpha,
              double theta) {
  size_t h = f.size();
  // Each step reads only the row before it, so two rolling rows stand in for
  // the (i + h) x 5 matrix this used to allocate and copy the history into.
  std::array<double, 5> prev, cur;
  std::copy_n(states.row(i - 1).data(), prev.size(), prev.data());
  for (size_t j = 0; j < h; ++j) {
    update_step(prev.data(), cur.data(), i + j, model_type, alpha, theta,
                double{}, true);
    f[j] = cur[4];
    prev = cur;
  }
}

// Workhorse taking caller-owned scratch: denom and f must hold nmse entries.
// Only denom is re-zeroed here; f is fully written by forecast before it is
// read.
double calc_buf(const Eigen::Ref<const VectorXd> &y,
                Eigen::Ref<RowMajorMatrixXd> states, ModelType model_type,
                double initial_smoothed, double alpha, double theta,
                Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, size_t nmse,
                Eigen::Ref<VectorXd> denom, Eigen::Ref<VectorXd> f) {
  denom.setZero();
  auto init_states = init_state(y, model_type, initial_smoothed, alpha, theta);
  std::ranges::copy(init_states, states.row(0).begin());
  std::fill_n(amse.begin(), nmse, double{});
  e[0] = y[0] - states(0, 4);
  size_t n = y.size();
  for (size_t i = 1; i < n; ++i) {
    forecast(states, i, model_type, f, alpha, theta);
    if (std::abs(f[0] - NA) < TOL) {
      return NA;
    }
    e[i] = y[i] - f[0];
    for (size_t j = 0; j < nmse; ++j) {
      if (i + j < n) {
        denom[j] += 1.0;
        double tmp = y[i + j] - f[j];
        amse[j] = (amse[j] * (denom[j] - 1.0) + tmp * tmp) / denom[j];
      }
    }
    update(states, i, model_type, alpha, theta, y[i], false);
  }
  double mean_y = y.array().abs().mean();
  if (mean_y < TOL) {
    mean_y = TOL;
  }
  return e.tail(e.size() - 3).array().square().sum() / mean_y;
}

// Allocating version (for the public calc API where the scratch isn't reused)
double calc(const Eigen::Ref<const VectorXd> &y,
            Eigen::Ref<RowMajorMatrixXd> states, ModelType model_type,
            double initial_smoothed, double alpha, double theta,
            Eigen::Ref<VectorXd> e, Eigen::Ref<VectorXd> amse, size_t nmse) {
  VectorXd denom(nmse);
  VectorXd f(nmse);
  return calc_buf(y, states, model_type, initial_smoothed, alpha, theta, e,
                  amse, nmse, denom, f);
}

std::tuple<VectorXd, VectorXd, RowMajorMatrixXd, double>
pegels_resid(const Eigen::Ref<const VectorXd> &y, ModelType model_type,
             double initial_smoothed, double alpha, double theta, size_t nmse) {
  RowMajorMatrixXd states = RowMajorMatrixXd::Zero(y.size(), 5);
  VectorXd e = VectorXd::Zero(y.size());
  VectorXd amse = VectorXd::Zero(nmse);
  double mse = calc(y, states, model_type, initial_smoothed, alpha, theta, e,
                    amse, nmse);
  if (!std::isnan(mse) && std::abs(mse + 99999) < 1e-7) {
    mse = std::numeric_limits<double>::quiet_NaN();
  }
  return {amse, e, states, mse};
}

// Buffers shared by the objective evaluations of a single optimize() call.
// Function-local, never static: fits may run concurrently.
struct Scratch {
  RowMajorMatrixXd states;
  VectorXd e, amse, denom, f;

  Scratch(Eigen::Index n, size_t nmse)
      : states(n, 5), e(n), amse(nmse), denom(nmse), f(nmse) {}
};

double target_fn(const Eigen::Ref<const VectorXd> &params, Scratch &ws,
                 double init_level, double init_alpha, double init_theta,
                 bool opt_level, bool opt_alpha, bool opt_theta,
                 const Eigen::Ref<const VectorXd> &y, ModelType model_type,
                 size_t nmse) {
  size_t j = 0;
  double level, alpha, theta;
  if (opt_level) {
    level = params[j++];
  } else {
    level = init_level;
  }
  if (opt_alpha) {
    alpha = params[j++];
  } else {
    alpha = init_alpha;
  }
  if (opt_theta) {
    theta = params[j++];
  } else {
    theta = init_theta;
  }
  double mse = calc_buf(y, ws.states, model_type, level, alpha, theta, ws.e,
                        ws.amse, nmse, ws.denom, ws.f);
  mse = std::max(mse, -1e10);
  if (std::isnan(mse) || std::abs(mse + 99999) < 1e-7) {
    mse = -std::numeric_limits<double>::infinity();
  }
  return mse;
}

nm::OptimResult optimize(const Eigen::Ref<const VectorXd> &x0,
                         const Eigen::Ref<const VectorXd> &lower,
                         const Eigen::Ref<const VectorXd> &upper,
                         double init_level, double init_alpha,
                         double init_theta, bool opt_level, bool opt_alpha,
                         bool opt_theta, const Eigen::Ref<const VectorXd> &y,
                         ModelType model_type, size_t nmse) {
  double init_step = 0.05;
  double zero_pert = 1e-4;
  double alpha = 1.0;
  double gamma = 2.0;
  double rho = 0.5;
  double sigma = 0.5;
  int max_iter = 1'000;
  double tol_std = 1e-4;
  bool adaptive = true;
  Scratch ws(y.size(), nmse);
  return nm::NelderMead(target_fn, x0, lower, upper, init_step, zero_pert,
                        alpha, gamma, rho, sigma, max_iter, tol_std, adaptive,
                        ws, init_level, init_alpha, init_theta, opt_level,
                        opt_alpha, opt_theta, y, model_type, nmse);
}

double target_fn_dist(const Eigen::Ref<const VectorXd> &params, Scratch &ws,
                      double init_level, double init_alpha, double init_theta,
                      bool opt_level, bool opt_alpha, bool opt_theta,
                      const Eigen::Ref<const VectorXd> &y,
                      ModelType model_type, size_t nmse,
                      dist::Distribution distribution) {
  size_t j = 0;
  double level = opt_level ? params[j++] : init_level;
  double alpha = opt_alpha ? params[j++] : init_alpha;
  double theta = opt_theta ? params[j++] : init_theta;
  // e is only read past the guard below, by when calc_buf has written all of it
  VectorXd &e = ws.e;
  double mse = calc_buf(y, ws.states, model_type, level, alpha, theta, e,
                        ws.amse, nmse, ws.denom, ws.f);
  if (std::isnan(mse) || std::abs(mse + 99999) < 1e-7)
    return std::numeric_limits<double>::infinity();

  int n_eff = static_cast<int>(y.size()) - 3;
  const double *e3 = e.data() + 3;
  int n_total = static_cast<int>(params.size());
  switch (distribution) {
  case dist::Distribution::Laplace:
    return dist::negloglik_laplace(e3, n_eff);
  case dist::Distribution::StudentT:
    return dist::negloglik_t(e3, n_eff, params(n_total - 2),
                             params(n_total - 1));
  case dist::Distribution::SkewNormal:
    return dist::negloglik_skewnorm(e3, n_eff, params(n_total - 2),
                                    params(n_total - 1));
  case dist::Distribution::GED:
    return dist::negloglik_ged(e3, n_eff, params(n_total - 2),
                               params(n_total - 1));
  default:
    return std::numeric_limits<double>::infinity();
  }
}

nm::OptimResult optimize_dist(const Eigen::Ref<const VectorXd> &x0,
                              const Eigen::Ref<const VectorXd> &lower,
                              const Eigen::Ref<const VectorXd> &upper,
                              double init_level, double init_alpha,
                              double init_theta, bool opt_level, bool opt_alpha,
                              bool opt_theta, const Eigen::Ref<const VectorXd> &y,
                              ModelType model_type, size_t nmse,
                              dist::Distribution distribution) {
  Scratch ws(y.size(), nmse);
  return nm::NelderMead(target_fn_dist, x0, lower, upper, 0.05, 1e-4, 1.0,
                        2.0, 0.5, 0.5, 1000, 1e-4, true, ws, init_level,
                        init_alpha, init_theta, opt_level, opt_alpha, opt_theta,
                        y, model_type, nmse, distribution);
}

void init(py::module_ &m) {
  py::module_ theta = m.def_submodule("theta");
  theta.attr("HUGE_N") = HUGE_N;
  theta.attr("NA") = NA;
  theta.attr("TOL") = TOL;
  py::enum_<ModelType>(theta, "ModelType")
      .value("STM", ModelType::STM)
      .value("OTM", ModelType::OTM)
      .value("DSTM", ModelType::DSTM)
      .value("DOTM", ModelType::DOTM);
  theta.def("init_state", &init_state);
  theta.def("calc", &calc);
  theta.def("forecast", &forecast);
  theta.def("update", &update);
  theta.def("optimize", &optimize);
  theta.def("pegels_resid", &pegels_resid);
  // Distribution enum: alias from ets (already registered there; re-registering
  // would throw). ets::init runs before theta::init in statsforecast.cpp.
  theta.attr("Distribution") = m.attr("ets").attr("Distribution");
  theta.def("optimize_dist", &optimize_dist);
}

} // namespace theta
