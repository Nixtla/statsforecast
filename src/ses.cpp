#include <cmath>
#include <stdexcept>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace ses {
namespace py = pybind11;
using Eigen::VectorXd;

double ses_sse(double alpha, const Eigen::Ref<const VectorXd> &x) {
  double complement = 1.0 - alpha;
  double forecast = x[0];
  double sse = 0.0;
  for (Eigen::Index i = 1; i < x.size(); ++i) {
    forecast = alpha * x[i - 1] + complement * forecast;
    double err = x[i] - forecast;
    sse += err * err;
  }
  return sse;
}

std::tuple<double, VectorXd> ses_forecast(const Eigen::Ref<const VectorXd> &x,
                                          double alpha) {
  double complement = 1.0 - alpha;
  VectorXd fitted(x.size());
  fitted[0] = x[0];
  Eigen::Index j = 0;
  for (Eigen::Index i = 1; i < x.size(); ++i) {
    fitted[i] = alpha * x[j] + complement * fitted[j];
    j += 1;
  }
  double forecast = alpha * x[j] + complement * fitted[j];
  fitted[0] = std::numeric_limits<double>::quiet_NaN();
  return {forecast, fitted};
}

VectorXd expand_fitted_demand(const Eigen::Ref<const VectorXd> &fitted,
                              const Eigen::Ref<const VectorXd> &y) {
  VectorXd out(y.size());
  out[0] = std::numeric_limits<double>::quiet_NaN();
  Eigen::Index fitted_idx = 0;
  Eigen::Index fitted_size = fitted.size();
  for (Eigen::Index i = 1; i < y.size(); ++i) {
    if (y[i - 1] > 0) {
      fitted_idx += 1;
      if (fitted_idx >= fitted_size) {
        throw std::out_of_range(
            "expand_fitted_demand: fitted_idx (" +
            std::to_string(fitted_idx) +
            ") exceeds fitted size (" +
            std::to_string(fitted_size) + ")");
      }
      out[i] = fitted[fitted_idx];
    } else if (fitted_idx > 0) {
      out[i] = out[i - 1];
    } else {
      out[i] = y[i - 1];
    }
  }
  return out;
}

VectorXd expand_fitted_intervals(const Eigen::Ref<const VectorXd> &fitted,
                                 const Eigen::Ref<const VectorXd> &y) {
  VectorXd out(y.size());
  out[0] = std::numeric_limits<double>::quiet_NaN();
  Eigen::Index fitted_idx = 0;
  Eigen::Index fitted_size = fitted.size();
  for (Eigen::Index i = 1; i < y.size(); ++i) {
    if (y[i - 1] != 0) {
      fitted_idx += 1;
      if (fitted_idx >= fitted_size) {
        throw std::out_of_range(
            "expand_fitted_intervals: fitted_idx (" +
            std::to_string(fitted_idx) +
            ") exceeds fitted size (" +
            std::to_string(fitted_size) + ")");
      }
      if (fitted[fitted_idx] == 0) {
        out[i] = 1.0;
      } else {
        out[i] = fitted[fitted_idx];
      }
    } else if (fitted_idx > 0) {
      out[i] = out[i - 1];
    } else {
      out[i] = 1.0;
    }
  }
  return out;
}

void init(py::module_ &m) {
  py::module_ ses_mod = m.def_submodule("ses");
  ses_mod.def("ses_sse", &ses_sse);
  ses_mod.def("ses_forecast", &ses_forecast);
  ses_mod.def("expand_fitted_demand", &expand_fitted_demand);
  ses_mod.def("expand_fitted_intervals", &expand_fitted_intervals);
}

} // namespace ses
