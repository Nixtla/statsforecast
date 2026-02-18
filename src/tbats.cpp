#include <cmath>
#include <stdexcept>
#include <tuple>

#include <Eigen/Dense>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Named tbats_ns (not tbats) to avoid collision with the pybind11 submodule name
namespace tbats_ns {
namespace py = pybind11;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using RowMajorMatrixXd =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

RowMajorMatrixXd makeFMatrix(py::object phi_obj, int tau, double alpha,
                              py::object beta_obj,
                              py::object ar_coeffs_obj,
                              py::object ma_coeffs_obj,
                              const Eigen::Ref<const RowMajorMatrixXd> &gamma_bold,
                              const Eigen::Ref<const VectorXd> &seasonal_periods,
                              const Eigen::Ref<const VectorXd> &k_vector) {
  bool has_phi = !phi_obj.is_none();
  bool has_beta = !beta_obj.is_none();
  bool has_ar = !ar_coeffs_obj.is_none();
  bool has_ma = !ma_coeffs_obj.is_none();

  double phi = has_phi ? phi_obj.cast<double>() : 0.0;
  double beta = has_beta ? beta_obj.cast<double>() : 0.0;
  VectorXd ar_coeffs, ma_coeffs;
  int p = 0, q = 0;
  if (has_ar) {
    ar_coeffs = ar_coeffs_obj.cast<VectorXd>();
    p = ar_coeffs.size();
  }
  if (has_ma) {
    ma_coeffs = ma_coeffs_obj.cast<VectorXd>();
    q = ma_coeffs.size();
  }

  // Compute total dimension
  int dim = 1; // level
  if (has_phi)
    dim += 1; // beta/trend row adds a column
  // Wait, phi adds a column to the alpha row, but beta adds a row
  // Let me compute correctly:
  // Rows: 1 (level) + has_beta (trend) + tau (seasonal) + p (AR) + q (MA)
  // Cols: same (square matrix)
  int total_dim = 1 + (has_beta ? 1 : 0) + tau + p + q;
  RowMajorMatrixXd F = RowMajorMatrixXd::Zero(total_dim, total_dim);

  int row_offset = 0;
  int col_offset = 0;

  // Alpha row (row 0)
  F(0, 0) = 1.0; // level
  col_offset = 1;
  if (has_phi) {
    F(0, col_offset) = phi;
    col_offset++;
  }
  // Skip tau seasonal columns (zeros)
  col_offset = 1 + (has_beta ? 1 : 0) + tau;
  if (has_ar) {
    for (int j = 0; j < p; ++j) {
      F(0, col_offset + j) = alpha * ar_coeffs[j];
    }
    col_offset += p;
  }
  if (has_ma) {
    for (int j = 0; j < q; ++j) {
      F(0, col_offset + j) = alpha * ma_coeffs[j];
    }
  }
  row_offset = 1;

  // Beta row (row 1, if has_beta)
  if (has_beta) {
    F(row_offset, 0) = 0.0;
    F(row_offset, 1) = phi;
    int c = 1 + 1 + tau; // skip level, phi col, tau seasonal
    if (has_ar) {
      for (int j = 0; j < p; ++j) {
        F(row_offset, c + j) = beta * ar_coeffs[j];
      }
      c += p;
    }
    if (has_ma) {
      for (int j = 0; j < q; ++j) {
        F(row_offset, c + j) = beta * ma_coeffs[j];
      }
    }
    row_offset++;
  }

  // Seasonal rows (tau rows)
  // First fill with the A matrix (trigonometric)
  int seasonal_col_start = 1 + (has_beta ? 1 : 0);
  int pos = 0;
  for (Eigen::Index s = 0; s < k_vector.size(); ++s) {
    int k = (int)k_vector[s];
    double period = seasonal_periods[s];
    for (int j = 0; j < k; ++j) {
      double t = 2.0 * M_PI * (j + 1) / period;
      double ct = std::cos(t);
      double st = std::sin(t);
      int r = row_offset + pos + j;
      int rk = row_offset + pos + k + j;
      int c = seasonal_col_start + pos + j;
      int ck = seasonal_col_start + pos + k + j;
      // cos/sin block
      F(r, c) = ct;
      F(r, ck) = st;
      F(rk, c) = -st;
      F(rk, ck) = ct;
    }
    pos += 2 * k;
  }

  // gamma_bold^T * ar/ma_coeffs for seasonal rows
  // In Python: B = np.dot(np.transpose(gamma_bold), varphi)
  // gamma_bold is (1, tau), gamma_bold.T is (tau, 1), varphi is (1, p)
  // B = (tau, 1) @ (1, p) = (tau, p), so B[i,j] = gamma_bold[0,i] * ar_coeffs[j]
  int arma_col_start = seasonal_col_start + tau;
  if (has_ar) {
    for (int i = 0; i < tau; ++i) {
      for (int j = 0; j < p; ++j) {
        F(row_offset + i, arma_col_start + j) = gamma_bold(0, i) * ar_coeffs[j];
      }
    }
    arma_col_start += p;
  }
  if (has_ma) {
    for (int i = 0; i < tau; ++i) {
      for (int j = 0; j < q; ++j) {
        F(row_offset + i, arma_col_start + j) = gamma_bold(0, i) * ma_coeffs[j];
      }
    }
  }
  row_offset += tau;

  // AR rows
  if (has_ar) {
    int ar_col_start = 1 + (has_beta ? 1 : 0) + tau;
    // First row: ar_coeffs
    for (int j = 0; j < p; ++j) {
      F(row_offset, ar_col_start + j) = ar_coeffs[j];
    }
    // Identity shifted
    for (int i = 1; i < p; ++i) {
      F(row_offset + i, ar_col_start + i - 1) = 1.0;
    }
    // MA part in AR block
    if (has_ma) {
      int ma_col = ar_col_start + p;
      for (int j = 0; j < q; ++j) {
        F(row_offset, ma_col + j) = ma_coeffs[j];
      }
    }
    row_offset += p;
  }

  // MA rows
  if (has_ma) {
    int ma_col_start = 1 + (has_beta ? 1 : 0) + tau + p;
    // Identity shifted (skip first row which is zeros)
    for (int i = 1; i < q; ++i) {
      F(row_offset + i, ma_col_start + i - 1) = 1.0;
    }
    row_offset += q;
  }

  return F;
}

std::tuple<RowMajorMatrixXd, RowMajorMatrixXd, RowMajorMatrixXd>
calcFaster(const Eigen::Ref<const VectorXd> &y_trans,
           const Eigen::Ref<const VectorXd> &w_transpose,
           const Eigen::Ref<const RowMajorMatrixXd> &g,
           const Eigen::Ref<const RowMajorMatrixXd> &F,
           const Eigen::Ref<const VectorXd> &x_nought) {
  Eigen::Index n = y_trans.size();
  Eigen::Index state_dim = x_nought.size();

  RowMajorMatrixXd yhat = RowMajorMatrixXd::Zero(1, n);
  RowMajorMatrixXd e = RowMajorMatrixXd::Zero(1, n);
  RowMajorMatrixXd x = RowMajorMatrixXd::Zero(n, state_dim);

  yhat(0, 0) = w_transpose.dot(x_nought);
  e(0, 0) = y_trans[0] - yhat(0, 0);
  x.row(0) = (F * x_nought + g.col(0) * e(0, 0)).transpose();

  for (Eigen::Index j = 1; j < n; ++j) {
    yhat(0, j) = w_transpose.dot(x.row(j - 1).transpose());
    e(0, j) = y_trans[j] - yhat(0, j);
    x.row(j) = (F * x.row(j - 1).transpose() + g.col(0) * e(0, j)).transpose();
  }

  return {yhat, e, x};
}

void init(py::module_ &m) {
  py::module_ tbats_mod = m.def_submodule("tbats");
  tbats_mod.def("makeFMatrix", &makeFMatrix);
  tbats_mod.def("calcFaster", &calcFaster);
}

} // namespace tbats_ns
