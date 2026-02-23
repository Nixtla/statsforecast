#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace mfles {
namespace py = pybind11;
using Eigen::VectorXd;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMajorMatrixXd get_basis(const Eigen::Ref<const VectorXd> &y_in,
                           int n_changepoints, double decay,
                           int gradient_strategy) {
  Eigen::Index n = y_in.size();
  if (n_changepoints < 1) {
    RowMajorMatrixXd result(n, 1);
    for (Eigen::Index i = 0; i < n; ++i) {
      result(i, 0) = (double)i;
    }
    return result;
  }

  VectorXd y = y_in;
  double y0 = y[0];
  y.array() -= y0;

  double initial_point = y[0];
  double final_point = y[n - 1];
  double mean_y = y.mean();

  RowMajorMatrixXd changepoints(n, n_changepoints + 1);

  // Compute gradients if needed
  VectorXd gradients;
  if (gradient_strategy) {
    gradients.resize(n - 1);
    for (Eigen::Index i = 0; i < n - 1; ++i) {
      gradients[i] = std::fabs(y[i] - y[i + 1]);
    }
  }

  // Collect array splits and their lengths
  struct SplitInfo {
    Eigen::Index length;
    double last_value;
  };
  std::vector<SplitInfo> splits(n_changepoints);

  // Pre-sort gradient indices once (reused across all changepoints)
  std::vector<Eigen::Index> grad_sorted_indices;
  std::vector<Eigen::Index> grad_filtered;
  if (gradient_strategy) {
    Eigen::Index grad_n = gradients.size();
    grad_sorted_indices.resize(grad_n);
    std::iota(grad_sorted_indices.begin(), grad_sorted_indices.end(), 0);
    std::sort(grad_sorted_indices.begin(), grad_sorted_indices.end(),
              [&gradients](Eigen::Index a, Eigen::Index b) {
                return gradients[a] > gradients[b];
              });
    // Filter: keep indices whose value falls in (0.1*len, 0.9*len)
    Eigen::Index low_bound = static_cast<Eigen::Index>(0.1 * grad_n);
    Eigen::Index high_bound = static_cast<Eigen::Index>(0.9 * grad_n);
    for (auto ci : grad_sorted_indices) {
      if (ci > low_bound && ci < high_bound) {
        grad_filtered.push_back(ci);
      }
    }
  }

  // Work on a copy for the non-gradient strategy
  VectorXd y_work = y;
  for (int idx = 0; idx < n_changepoints; ++idx) {
    int i = n_changepoints - idx; // goes from n_changepoints down to 1
    Eigen::Index split_point;
    if (gradient_strategy) {
      if (i - 1 >= static_cast<int>(grad_filtered.size())) {
        throw std::runtime_error(
            "get_basis: not enough gradient changepoint candidates ("
            + std::to_string(grad_filtered.size())
            + ") for n_changepoints=" + std::to_string(n_changepoints));
      }
      split_point = grad_filtered[i - 1];
      if (split_point < 1) {
        throw std::runtime_error(
            "get_basis: gradient split_point is 0, cannot index y[-1]");
      }
      splits[idx].length = split_point;
      splits[idx].last_value = y[split_point - 1];
    } else {
      split_point = y_work.size() / i;
      splits[idx].length = split_point;
      splits[idx].last_value = y_work[split_point - 1];
      // Shift y_work forward
      VectorXd temp = y_work.tail(y_work.size() - split_point);
      y_work = temp;
    }
  }

  Eigen::Index len_splits = 0;
  for (int i = 0; i < n_changepoints; ++i) {
    if (gradient_strategy) {
      len_splits = splits[i].length;
    } else {
      len_splits += splits[i].length;
    }
    double moving_point = splits[i].last_value;

    // Compute left basis: linspace(initial_point, moving_point, len_splits)
    for (Eigen::Index j = 0; j < len_splits; ++j) {
      if (len_splits > 1) {
        changepoints(j, i) =
            initial_point + (moving_point - initial_point) * j / (len_splits - 1);
      } else {
        changepoints(j, i) = initial_point;
      }
    }

    // Compute end_point based on decay
    double end_point;
    bool decay_is_none = std::isnan(decay); // Using NaN to represent None
    if (decay_is_none) {
      end_point = final_point;
    } else if (decay == -1.0) {
      double dd = moving_point * moving_point;
      if (mean_y != 0.0) {
        dd /= (mean_y * mean_y);
      }
      if (dd > 0.99)
        dd = 0.99;
      if (dd < 0.001)
        dd = 0.001;
      end_point = moving_point - ((moving_point - final_point) * (1.0 - dd));
    } else {
      end_point = moving_point - ((moving_point - final_point) * (1.0 - decay));
    }

    // Compute right basis: linspace(moving_point, end_point, n - len_splits + 1)
    // Then append right_basis[1:] to left_basis
    Eigen::Index right_n = n - len_splits + 1;
    for (Eigen::Index j = 1; j < right_n; ++j) {
      double val;
      if (right_n > 1) {
        val = moving_point + (end_point - moving_point) * j / (right_n - 1);
      } else {
        val = moving_point;
      }
      changepoints(len_splits + j - 1, i) = val;
    }
  }

  // Last column: ones
  changepoints.col(n_changepoints).setOnes();

  return changepoints;
}

VectorXd siegel_repeated_medians(const Eigen::Ref<const VectorXd> &x,
                                 const Eigen::Ref<const VectorXd> &y) {
  Eigen::Index n = y.size();
  if (n < 2) {
    throw std::invalid_argument(
        "siegel_repeated_medians requires at least 2 data points");
  }
  VectorXd slopes(n);
  std::vector<double> slopes_sub(n - 1);

  for (Eigen::Index i = 0; i < n; ++i) {
    Eigen::Index k = 0;
    for (Eigen::Index j = 0; j < n; ++j) {
      if (i == j)
        continue;
      double xd = x[j] - x[i];
      double slope;
      if (xd == 0.0) {
        slope = 0.0;
      } else {
        slope = (y[j] - y[i]) / xd;
      }
      slopes_sub[k] = slope;
      k++;
    }
    // Compute median of slopes_sub (size n-1)
    Eigen::Index m = n - 1;
    std::nth_element(slopes_sub.begin(), slopes_sub.begin() + m / 2,
                     slopes_sub.begin() + m);
    if (m % 2 == 1) {
      // Odd count: middle element is the median
      slopes[i] = slopes_sub[m / 2];
    } else {
      // Even count: average of two middle elements
      double mid = slopes_sub[m / 2];
      double left =
          *std::max_element(slopes_sub.begin(), slopes_sub.begin() + m / 2);
      slopes[i] = (left + mid) / 2.0;
    }
  }

  // Compute median of slopes
  std::vector<double> slopes_vec(slopes.data(), slopes.data() + n);
  std::nth_element(slopes_vec.begin(), slopes_vec.begin() + n / 2,
                   slopes_vec.end());
  double median_slope;
  if (n % 2 == 1) {
    median_slope = slopes_vec[n / 2];
  } else {
    double mid = slopes_vec[n / 2];
    double left =
        *std::max_element(slopes_vec.begin(), slopes_vec.begin() + n / 2);
    median_slope = (left + mid) / 2.0;
  }

  // Compute intercepts and their median
  VectorXd ints = y - (slopes.array() * x.array()).matrix();
  std::vector<double> ints_vec(ints.data(), ints.data() + n);
  std::nth_element(ints_vec.begin(), ints_vec.begin() + n / 2, ints_vec.end());
  double median_int;
  if (n % 2 == 1) {
    median_int = ints_vec[n / 2];
  } else {
    double mid = ints_vec[n / 2];
    double left =
        *std::max_element(ints_vec.begin(), ints_vec.begin() + n / 2);
    median_int = (left + mid) / 2.0;
  }

  return x.array() * median_slope + median_int;
}

void init(py::module_ &m) {
  py::module_ mfles_mod = m.def_submodule("mfles");
  mfles_mod.def("get_basis", &get_basis,
                py::call_guard<py::gil_scoped_release>());
  mfles_mod.def("siegel_repeated_medians", &siegel_repeated_medians,
                py::call_guard<py::gil_scoped_release>());
}

} // namespace mfles
