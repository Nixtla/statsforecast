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
  if (x.size() < 2) {
    return 0.0;
  }
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
  if (x.size() < 1) {
    throw std::invalid_argument(
        "ses_forecast requires at least 1 data point");
  }
  if (x.size() == 1) {
    VectorXd fitted(1);
    fitted[0] = std::numeric_limits<double>::quiet_NaN();
    return {x[0], fitted};
  }
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

double golden_section_ses(const Eigen::Ref<const VectorXd> &x,
                          double lower = 0.1, double upper = 0.3) {
  const double gr = (std::sqrt(5.0) + 1.0) / 2.0;
  double a = lower, b = upper;
  double c = b - (b - a) / gr;
  double d = a + (b - a) / gr;
  double fc = ses_sse(c, x);
  double fd = ses_sse(d, x);
  while (std::abs(b - a) >= 1e-12) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - (b - a) / gr;
      fc = ses_sse(c, x);
    } else if (fd < fc) {
      a = c;
      c = d;
      fc = fd;
      d = a + (b - a) / gr;
      fd = ses_sse(d, x);
    } else {
      break;
    }
  }
  return (b + a) / 2.0;
}

double chunk_forecast(const Eigen::Ref<const VectorXd> &y,
                      Eigen::Index aggregation_level) {
  if (aggregation_level <= 0) {
    throw std::invalid_argument(
        "chunk_forecast: aggregation_level must be >= 1");
  }
  Eigen::Index n = y.size();
  Eigen::Index lost = n % aggregation_level;
  Eigen::Index n_cut = n - lost;
  if (n_cut < aggregation_level) {
    return y[n - 1];
  }
  Eigen::Index n_chunks = n_cut / aggregation_level;
  VectorXd agg_sums(n_chunks);
  for (Eigen::Index i = 0; i < n_chunks; ++i) {
    double s = 0.0;
    Eigen::Index base = lost + i * aggregation_level;
    for (Eigen::Index j = 0; j < aggregation_level; ++j) {
      s += y[base + j];
    }
    agg_sums[i] = s;
  }
  if (n_chunks <= 1) {
    return agg_sums[0];
  }
  double alpha = golden_section_ses(agg_sums);
  auto [forecast, fitted] = ses_forecast(agg_sums, alpha);
  return forecast;
}

VectorXd adida_fitted_vals(const Eigen::Ref<const VectorXd> &y,
                           const Eigen::Ref<const Eigen::VectorXi> &agg_levels) {
  Eigen::Index n = y.size() - 1;
  VectorXd sums_fitted(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    sums_fitted[i] = chunk_forecast(y.head(i + 1), agg_levels[i]);
  }
  return sums_fitted;
}

VectorXd imapa_fitted_vals(const Eigen::Ref<const VectorXd> &y) {
  Eigen::Index n = y.size();
  VectorXd fitted_vals(n);
  fitted_vals[0] = std::numeric_limits<double>::quiet_NaN();
  for (Eigen::Index i = 1; i < n; ++i) {
    // count non-zero elements and compute mean interval in a single pass
    int nonzero_count = 0;
    double intervals_sum = 0.0;
    Eigen::Index prev = 0;
    for (Eigen::Index j = 0; j < i; ++j) {
      if (y[j] != 0.0) {
        ++nonzero_count;
        intervals_sum += static_cast<double>(j + 1 - prev);
        prev = j + 1;
      }
    }
    if (nonzero_count == 0) {
      fitted_vals[i] = 0.0;
      continue;
    }
    double mean_interval = intervals_sum / nonzero_count;
    Eigen::Index max_agg =
        std::max(static_cast<Eigen::Index>(1),
                 static_cast<Eigen::Index>(std::round(mean_interval)));
    max_agg = std::min(max_agg, i);
    // average over aggregation levels
    double forecast_sum = 0.0;
    int count = 0;
    auto prefix = y.head(i);
    for (Eigen::Index agg_level = 1; agg_level <= max_agg; ++agg_level) {
      double fcst = chunk_forecast(prefix, agg_level);
      forecast_sum += fcst / static_cast<double>(agg_level);
      ++count;
    }
    fitted_vals[i] = forecast_sum / count;
  }
  return fitted_vals;
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
  ses_mod.def("ses_sse", &ses_sse,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("ses_forecast", &ses_forecast,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("golden_section_ses", &golden_section_ses,
              py::arg("x"), py::arg("lower") = 0.1, py::arg("upper") = 0.3,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("chunk_forecast", &chunk_forecast,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("adida_fitted_vals", &adida_fitted_vals,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("imapa_fitted_vals", &imapa_fitted_vals,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("expand_fitted_demand", &expand_fitted_demand,
              py::call_guard<py::gil_scoped_release>());
  ses_mod.def("expand_fitted_intervals", &expand_fitted_intervals,
              py::call_guard<py::gil_scoped_release>());
}

} // namespace ses
