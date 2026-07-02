#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace ucm {
namespace py = pybind11;
using Eigen::MatrixXd;
using Eigen::VectorXd;

double kalman_filter_core(const Eigen::Ref<const VectorXd> &y,
                          const Eigen::Ref<const MatrixXd> &Z,
                          const Eigen::Ref<const MatrixXd> &T,
                          const Eigen::Ref<const MatrixXd> &R,
                          const Eigen::Ref<const MatrixXd> &Q, double H,
                          const Eigen::Ref<const VectorXd> &a0,
                          const Eigen::Ref<const MatrixXd> &P0, bool collect,
                          VectorXd &a_filt, MatrixXd &P_filt,
                          VectorXd &one_step_pred) {
  const Eigen::Index n = y.size();
  const VectorXd z = Z.row(0).transpose();
  const MatrixXd RQRt = R * Q * R.transpose();

  VectorXd a_pred = a0;
  MatrixXd P_pred = P0;

  a_filt = a_pred;
  P_filt = P_pred;

  double loglik = 0.0;
  const double log_2pi = std::log(2.0 * M_PI);

  for (Eigen::Index t = 0; t < n; ++t) {
    // Forecast for the current observation.
    double y_hat = z.dot(a_pred);
    if (collect) {
      one_step_pred[t] = y_hat;
    }
    double v = y[t] - y_hat;
    VectorXd Pz = P_pred * z;
    double F = z.dot(Pz) + H;

    // Update step (skip if F is not usable).
    if (std::isfinite(F) && F > 0.0) {
      loglik += -0.5 * (log_2pi + std::log(F) + v * v / F);
      VectorXd K = Pz / F;
      a_filt = a_pred + K * v;
      P_filt = P_pred - K * Pz.transpose();
    } else {
      a_filt = a_pred;
      P_filt = P_pred;
    }

    // Predict the next state.
    a_pred = T * a_filt;
    P_pred = T * P_filt * T.transpose() + RQRt;
  }

  return loglik;
}

// Scalar log-likelihood for the optimizer hot path.
double loglik(const Eigen::Ref<const VectorXd> &y,
              const Eigen::Ref<const MatrixXd> &Z,
              const Eigen::Ref<const MatrixXd> &T,
              const Eigen::Ref<const MatrixXd> &R,
              const Eigen::Ref<const MatrixXd> &Q, double H,
              const Eigen::Ref<const VectorXd> &a0,
              const Eigen::Ref<const MatrixXd> &P0) {
  VectorXd a_filt;
  MatrixXd P_filt;
  VectorXd dummy;
  return kalman_filter_core(y, Z, T, R, Q, H, a0, P0, false, a_filt, P_filt,
                            dummy);
}

// Full filter returning (loglik, a_filt, P_filt, one_step_pred) for the
// final fit.
std::tuple<double, VectorXd, MatrixXd, VectorXd>
filter(const Eigen::Ref<const VectorXd> &y,
       const Eigen::Ref<const MatrixXd> &Z,
       const Eigen::Ref<const MatrixXd> &T,
       const Eigen::Ref<const MatrixXd> &R,
       const Eigen::Ref<const MatrixXd> &Q, double H,
       const Eigen::Ref<const VectorXd> &a0,
       const Eigen::Ref<const MatrixXd> &P0) {
  VectorXd a_filt;
  MatrixXd P_filt;
  VectorXd one_step_pred(y.size());
  double ll =
      kalman_filter_core(y, Z, T, R, Q, H, a0, P0, true, a_filt, P_filt,
                         one_step_pred);
  return {ll, a_filt, P_filt, one_step_pred};
}

void init(py::module_ &m) {
  py::module_ ucm_mod = m.def_submodule("ucm");
  ucm_mod.def("loglik", &loglik, py::call_guard<py::gil_scoped_release>());
  ucm_mod.def("filter", &filter, py::call_guard<py::gil_scoped_release>());
}

} // namespace ucm
