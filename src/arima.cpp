#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>
#include <stdexcept>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace arima {
namespace py = pybind11;
namespace {
template <typename T> std::span<T> make_span(py::array_t<T> &array) {
  // py::ssize_t to size_t is safe because the size of an array is non-negative
  return {array.mutable_data(), static_cast<size_t>(array.size())};
}

template <typename T>
std::span<const T> make_cspan(const py::array_t<T> &array) {
  // py::ssize_t to size_t is safe because the size of an array is non-negative
  return {array.data(), static_cast<size_t>(array.size())};
}
} // namespace

void partrans(const uint32_t p, const std::span<const double> rawv,
              const std::span<double> newv) {
  assert(p <= rawv.size());
  assert(p <= newv.size());

  std::transform(rawv.begin(), rawv.begin() + p, newv.begin(),
                 [](double x) { return std::tanh(x); });

  std::vector<double> work(newv.begin(), newv.begin() + p);
  for (size_t j = 1; j < p; ++j) {
    for (size_t k = 0; k < j; ++k) {
      work[k] -= newv[j] * newv[j - k - 1];
    }
    std::copy(work.begin(), work.begin() + j, newv.begin());
  }
}

std::tuple<py::array_t<double>, py::array_t<double>>
arima_transpar(const py::array_t<double> params_inv,
               const py::array_t<int32_t> armav, bool trans) {
  assert(params_inv.ndim() == 1);
  assert(armav.ndim() == 1);

  const auto arma = make_cspan(armav);
  const auto params_in = make_cspan(params_inv);

  assert(arma.size() == 7);
  assert(arma[0] >= 0 && arma[1] >= 0 && arma[2] >= 0 && arma[3] >= 0 &&
         arma[4] >= 0 && arma[5] >= 0 && arma[6] >= 0);

  const auto mp = static_cast<size_t>(arma[0]);
  const auto mq = static_cast<size_t>(arma[1]);
  const auto msp = static_cast<size_t>(arma[2]);
  const auto msq = static_cast<size_t>(arma[3]);
  const auto ns = static_cast<size_t>(arma[4]);
  const uint32_t p = mp + ns * msp;
  const uint32_t q = mq + ns * msq;

  std::vector<double> params(params_in.begin(), params_in.end());

  py::array_t<double> phiv(p);
  py::array_t<double> thetav(q);
  const auto phi = make_span(phiv);
  const auto theta = make_span(thetav);

  if (trans) {
    if (mp > 0) {
      partrans(mp, params_in, params);
    }
    const uint32_t v = mp + mq;
    if (msp > 0) {
      partrans(msp, params_in.subspan(v), std::span(params).subspan(v));
    }
  }

  if (ns > 0) {
    std::copy(params.begin(), params.begin() + mp, phi.begin());
    std::fill(phi.begin() + mp, phi.begin() + p, 0.0);
    std::copy(params.begin() + mp, params.begin() + mp + mq, theta.begin());
    std::fill(theta.begin() + mq, theta.begin() + q, 0.0);

    for (size_t j = 0; j < msp; ++j) {
      phi[(j + 1) * ns - 1] += params[j + mp + mq];
      for (size_t i = 0; i < mp; ++i) {
        phi[(j + 1) * ns + i] -= params[i] * params[j + mp + mq];
      }
    }

    for (size_t j = 0; j < msq; ++j) {
      theta[(j + 1) * ns - 1] += params[j + mp + mq + msp];
      for (size_t i = 0; i < mq; ++i) {
        theta[(j + 1) * ns + i] += params[i + mp] * params[j + mp + mq + msp];
      }
    }
  } else {
    std::copy(params.begin(), params.begin() + mp, phi.begin());
    std::copy(params.begin() + mp, params.begin() + mp + mq, theta.begin());
  }

  return {phiv, thetav};
}

std::tuple<double, py::array_t<double>>
arima_css(const py::array_t<double> yv, const py::array_t<int32_t> armav,
          const py::array_t<double> phiv, const py::array_t<double> thetav) {
  assert(yv.ndim() == 1);
  assert(armav.ndim() == 1);
  assert(phiv.ndim() == 1);
  assert(thetav.ndim() == 1);

  // py::ssize_t to size_t is safe because the size of an array is non-negative
  const size_t n = static_cast<size_t>(yv.size());
  const size_t p = static_cast<size_t>(phiv.size());
  const size_t q = static_cast<size_t>(thetav.size());

  const auto y = make_cspan(yv);
  const auto arma = make_cspan(armav);
  const auto phi = make_cspan(phiv);
  const auto theta = make_cspan(thetav);

  assert(arma.size() == 7);
  assert(arma[0] >= 0 && arma[1] >= 0 && arma[2] >= 0 && arma[3] >= 0 &&
         arma[4] >= 0 && arma[5] >= 0 && arma[6] >= 0);

  const auto d = static_cast<size_t>(arma[5]);
  const auto D = static_cast<size_t>(arma[6]);
  const auto ncond =
      static_cast<size_t>(arma[0] + arma[5] + arma[4] * (arma[2] + arma[6]));
  uint32_t nu = 0;
  double ssq = 0.0;

  py::array_t<double> residv(n);
  const auto resid = make_span(residv);
  if (static_cast<size_t>(ncond) > resid.size()) {
     throw std::logic_error(
       "Internal error: resid length (" + std::to_string(resid.size()) +
       ") must be >= ncond (" + std::to_string(ncond) + ")"
     );
  }
  std::fill_n(resid.begin(), ncond, 0.0);

  std::vector<double> w(y.begin(), y.end());

  for (size_t _ = 0; _ < d; ++_) {
    for (size_t l = n - 1; l > 0; --l) {
      w[l] -= w[l - 1];
    }
  }

  const auto ns = static_cast<size_t>(arma[4]);
  for (size_t _ = 0; _ < D; ++_) {
    for (size_t l = n - 1; l >= ns; --l) {
      w[l] -= w[l - ns];
    }
  }

  for (size_t l = ncond; l < n; ++l) {
    double tmp = w[l];
    for (size_t j = 0; j < p; ++j) {
      tmp -= phi[j] * w[l - j - 1];
    }

    assert(l >= ncond);
    for (size_t j = 0; j < std::min(static_cast<size_t>(l - ncond), q); ++j) {
      if (l - j - 1 < 0) {
        continue;
      }
      tmp -= theta[j] * resid[l - j - 1];
    }

    resid[l] = tmp;
    if (!std::isnan(tmp)) {
      nu++;
      ssq += tmp * tmp;
    }
  }

  return {ssq / nu, residv};
}

std::tuple<double, py::array_t<double>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
arima_css_grad(const py::array_t<double> yv, const py::array_t<int32_t> armav,
               const py::array_t<double> phiv,
               const py::array_t<double> thetav) {
  assert(yv.ndim() == 1);
  assert(armav.ndim() == 1);
  assert(phiv.ndim() == 1);
  assert(thetav.ndim() == 1);

  const size_t n = static_cast<size_t>(yv.size());
  const size_t p = static_cast<size_t>(phiv.size());
  const size_t q = static_cast<size_t>(thetav.size());

  const auto y = make_cspan(yv);
  const auto arma = make_cspan(armav);
  const auto phi = make_cspan(phiv);
  const auto theta = make_cspan(thetav);

  assert(arma.size() == 7);
  assert(arma[0] >= 0 && arma[1] >= 0 && arma[2] >= 0 && arma[3] >= 0 &&
         arma[4] >= 0 && arma[5] >= 0 && arma[6] >= 0);

  const auto d = static_cast<size_t>(arma[5]);
  const auto D = static_cast<size_t>(arma[6]);
  const auto ns = static_cast<size_t>(arma[4]);
  const auto ncond =
      static_cast<size_t>(arma[0] + arma[5] + arma[4] * (arma[2] + arma[6]));
  uint32_t nu = 0;
  double ssq = 0.0;

  py::array_t<double> residv(n);
  const auto resid = make_span(residv);
  if (ncond > resid.size()) {
    throw std::logic_error(
        "Internal error: resid length (" + std::to_string(resid.size()) +
        ") must be >= ncond (" + std::to_string(ncond) + ")");
  }
  std::fill_n(resid.begin(), ncond, 0.0);

  // Forward pass: identical to arima_css
  std::vector<double> w(y.begin(), y.end());

  for (size_t _ = 0; _ < d; ++_) {
    for (size_t l = n - 1; l > 0; --l) {
      w[l] -= w[l - 1];
    }
  }

  for (size_t _ = 0; _ < D; ++_) {
    for (size_t l = n - 1; l >= ns; --l) {
      w[l] -= w[l - ns];
    }
  }

  for (size_t l = ncond; l < n; ++l) {
    double tmp = w[l];
    for (size_t j = 0; j < p; ++j) {
      tmp -= phi[j] * w[l - j - 1];
    }
    for (size_t j = 0; j < std::min(static_cast<size_t>(l - ncond), q); ++j) {
      tmp -= theta[j] * resid[l - j - 1];
    }
    resid[l] = tmp;
    if (!std::isnan(tmp)) {
      nu++;
      ssq += tmp * tmp;
    }
  }

  // Backward pass: reverse-mode autodiff
  py::array_t<double> d_phiv(p);
  py::array_t<double> d_thetav(q);
  py::array_t<double> d_yv(n);
  auto d_phi = make_span(d_phiv);
  auto d_theta = make_span(d_thetav);
  auto d_y = make_span(d_yv);

  std::fill(d_phi.begin(), d_phi.end(), 0.0);
  std::fill(d_theta.begin(), d_theta.end(), 0.0);

  std::vector<double> bar_e(n, 0.0);
  std::vector<double> bar_w(n, 0.0);

  // Step A: seed adjoint from ssq = sum(resid[l]^2)
  for (size_t l = ncond; l < n; ++l) {
    if (!std::isnan(resid[l])) {
      bar_e[l] = 2.0 * resid[l];
    }
  }

  // Step B: backward through residual recursion
  for (size_t l = n - 1; ; --l) {
    if (l < ncond)
      break;

    bar_w[l] += bar_e[l];

    for (size_t j = 0; j < p; ++j) {
      d_phi[j] -= bar_e[l] * w[l - j - 1];
      bar_w[l - j - 1] -= bar_e[l] * phi[j];
    }

    for (size_t j = 0; j < std::min(static_cast<size_t>(l - ncond), q); ++j) {
      d_theta[j] -= bar_e[l] * resid[l - j - 1];
      bar_e[l - j - 1] -= bar_e[l] * theta[j];
    }

    if (l == ncond)
      break;
  }

  // Step C: adjoint of seasonal differencing (reversed: forward order)
  for (size_t _ = 0; _ < D; ++_) {
    for (size_t l = ns; l < n; ++l) {
      bar_w[l - ns] -= bar_w[l];
    }
  }

  // Step D: adjoint of regular differencing (reversed: forward order)
  for (size_t _ = 0; _ < d; ++_) {
    for (size_t l = 1; l < n; ++l) {
      bar_w[l - 1] -= bar_w[l];
    }
  }

  std::copy(bar_w.begin(), bar_w.end(), d_y.begin());

  return {ssq / nu, residv, d_phiv, d_thetav, d_yv};
}

std::tuple<double, double, int>
arima_like(const py::array_t<double> yv, const py::array_t<double> phiv,
           const py::array_t<double> thetav, const py::array_t<double> deltav,
           py::array_t<double> av, py::array_t<double> Pv,
           py::array_t<double> Pnewv, uint32_t up, bool use_resid,
           py::array_t<double> rsResidv) {
  // py::ssize_t to size_t is safe because the size of an array is non-negative
  const size_t n = static_cast<size_t>(yv.size());
  const size_t d = static_cast<size_t>(deltav.size());
  const size_t rd = static_cast<size_t>(av.size());
  const size_t p = static_cast<size_t>(phiv.size());
  const size_t q = static_cast<size_t>(thetav.size());

  const auto y = make_cspan(yv);
  const auto phi = make_cspan(phiv);
  const auto theta = make_cspan(thetav);
  const auto delta = make_cspan(deltav);
  const auto a = make_span(av);
  const auto P = make_span(Pv);
  const auto Pnew = make_span(Pnewv);
  const auto rsResid = make_span(rsResidv);

  // asserts for copies at the end
  assert(Pnew.size() >= rd * rd);
  assert(P.size() >= Pnew.size());

  double ssq = 0.0;
  double sumlog = 0.0;

  assert(rd >= d);
  const size_t r = rd - d;
  uint32_t nu = 0;

  std::vector<double> anew(rd);
  std::vector<double> M(rd);
  std::vector<double> mm;
  if (d > 0) {
    mm.resize(rd * rd);
  }

  double tmp;
  for (size_t l = 0; l < n; ++l) {
    for (size_t i = 0; i < r; ++i) {
      if (i < r - 1) {
        tmp = a[i + 1];
      } else {
        tmp = 0.0;
      }
      if (i < p) {
        tmp += phi[i] * a[0];
      }
      anew[i] = tmp;
    }

    if (d > 0) {
      for (size_t i = r + 1; i < rd; ++i) {
        anew[i] = a[i - 1];
      }
      tmp = a[0];
      for (size_t i = 0; i < d; ++i) {
        tmp += delta[i] * a[r + i];
      }
      anew[r] = tmp;
    }

    if (l > up) {
      if (d == 0) {
        for (size_t i = 0; i < r; ++i) {
          const double vi = [&]() {
            if (i == 0) {
              return 1.0;
            } else if (i - 1 < q) {
              return theta[i - 1];
            } else {
              return 0.0;
            }
          }();

          for (size_t j = 0; j < r; ++j) {
            tmp = 0.0;
            if (j == 0) {
              tmp = vi;
            } else if (j - 1 < q) {
              tmp = vi * theta[j - 1];
            }
            if (i < p && j < p) {
              tmp += phi[i] * phi[j] * P[0];
            }
            if (i < r - 1 && j < r - 1) {
              tmp += P[i + 1 + r * (j + 1)];
            }
            if (i < p && j < r - 1) {
              tmp += phi[i] * P[j + 1];
            }
            if (j < p && i < r - 1) {
              tmp += phi[j] * P[i + 1];
            }
            Pnew[i + r * j] = tmp;
          }
        }

      } else {
        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            tmp = 0.0;
            if (i < p) {
              tmp += phi[i] * P[rd * j];
            }
            if (i < r - 1) {
              tmp += P[i + 1 + rd * j];
            }
            mm[i + rd * j] = tmp;
          }
        }

        for (size_t j = 0; j < rd; ++j) {
          tmp = P[rd * j];
          for (size_t k = 0; k < d; ++k) {
            tmp += delta[k] * P[r + k + rd * j];
          }
          mm[r + rd * j] = tmp;
        }

        for (size_t i = 1; i < d; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            mm[r + i + rd * j] = P[r + i - 1 + rd * j];
          }
        }

        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            tmp = 0.0;
            if (i < p) {
              tmp += phi[i] * mm[j];
            }
            if (i < r - 1) {
              tmp += mm[rd * (i + 1) + j];
            }
            Pnew[j + rd * i] = tmp;
          }
        }

        for (size_t j = 0; j < rd; ++j) {
          tmp = mm[j];
          for (size_t k = 0; k < d; ++k) {
            tmp += delta[k] * mm[rd * (r + k) + j];
          }
          Pnew[rd * r + j] = tmp;
        }

        for (size_t i = 1; i < d; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            Pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
          }
        }

        for (size_t i = 0; i < q + 1; ++i) {
          const double vi = i == 0 ? 1.0 : theta[i - 1];
          for (size_t j = 0; j < q + 1; ++j) {
            if (j == 0) {
              Pnew[i + rd * j] += vi;
            } else {
              Pnew[i + rd * j] += vi * theta[j - 1];
            }
          }
        }
      }
    }

    if (!std::isnan(y[l])) {
      double resid = y[l] - anew[0];
      for (size_t i = 0; i < d; ++i) {
        resid -= delta[i] * anew[r + i];
      }

      for (size_t i = 0; i < rd; ++i) {
        tmp = Pnew[i];
        for (size_t j = 0; j < d; ++j) {
          tmp += Pnew[i + (r + j) * rd] * delta[j];
        }
        M[i] = tmp;
      }

      double gain = M[0];
      for (size_t j = 0; j < d; ++j) {
        gain += delta[j] * M[r + j];
      }

      if (gain < 1e4) {
        nu++;
        if (gain == 0) {
          ssq = std::numeric_limits<double>::infinity();
        } else {
          ssq += resid * resid / gain;
        }
        sumlog += std::log(gain);
      }

      if (use_resid) {
        if (gain == 0) {
          rsResid[l] = std::numeric_limits<double>::infinity();
        } else {
          rsResid[l] = resid / std::sqrt(gain);
        }
      }

      if (gain == 0) {
        for (size_t i = 0; i < rd; ++i) {
          a[i] = std::numeric_limits<double>::infinity();
          for (size_t j = 0; j < rd; ++j) {
            Pnew[i + j * rd] = std::numeric_limits<double>::infinity();
          }
        }

      } else {
        for (size_t i = 0; i < rd; ++i) {
          a[i] = anew[i] + M[i] * resid / gain;
          for (size_t j = 0; j < rd; ++j) {
            P[i + j * rd] = Pnew[i + j * rd] - M[i] * M[j] / gain;
          }
        }
      }

    } else {
      std::copy(anew.begin(), anew.end(), a.begin());
      std::copy(Pnew.begin(), Pnew.begin() + rd * rd, P.begin());
      if (use_resid) {
        rsResid[l] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  return {ssq, sumlog, nu};
}

std::tuple<double, py::array_t<double>, py::array_t<double>,
           py::array_t<double>, py::array_t<double>>
arima_like_grad(const py::array_t<double> yv, const py::array_t<double> phiv,
                const py::array_t<double> thetav,
                const py::array_t<double> deltav, py::array_t<double> av,
                py::array_t<double> Pv, py::array_t<double> Pnewv,
                uint32_t up) {
  const size_t n = static_cast<size_t>(yv.size());
  const size_t d = static_cast<size_t>(deltav.size());
  const size_t rd = static_cast<size_t>(av.size());
  const size_t p = static_cast<size_t>(phiv.size());
  const size_t q = static_cast<size_t>(thetav.size());

  const auto y = make_cspan(yv);
  const auto phi = make_cspan(phiv);
  const auto theta = make_cspan(thetav);
  const auto delta = make_cspan(deltav);
  auto a = make_span(av);
  auto P = make_span(Pv);
  auto Pnew = make_span(Pnewv);

  assert(rd >= d);
  const size_t r = rd - d;

  // ===================== Forward pass with tape =====================
  // Per-step storage
  std::vector<double> a_tape(n * rd);
  std::vector<double> P_tape(n * rd * rd);
  std::vector<double> anew_tape(n * rd);
  std::vector<double> Pnew_tape(n * rd * rd);
  std::vector<double> M_tape(n * rd);
  std::vector<double> resid_tape(n);
  std::vector<double> gain_tape(n);
  std::vector<uint8_t> valid_tape(n, 0);    // contributed to likelihood
  std::vector<uint8_t> observed_tape(n, 0); // non-NaN observation
  std::vector<double> mm_tape;
  if (d > 0) {
    mm_tape.resize(n * rd * rd);
  }

  double ssq = 0.0;
  double sumlog = 0.0;
  uint32_t nu = 0;

  std::vector<double> anew(rd);
  std::vector<double> M(rd);
  std::vector<double> mm;
  if (d > 0) {
    mm.resize(rd * rd);
  }

  double tmp;
  for (size_t l = 0; l < n; ++l) {
    // Save a, P before prediction
    std::copy(a.begin(), a.begin() + rd, a_tape.begin() + l * rd);
    std::copy(P.begin(), P.begin() + rd * rd, P_tape.begin() + l * rd * rd);

    // --- State prediction ---
    for (size_t i = 0; i < r; ++i) {
      tmp = (i < r - 1) ? a[i + 1] : 0.0;
      if (i < p)
        tmp += phi[i] * a[0];
      anew[i] = tmp;
    }
    if (d > 0) {
      for (size_t i = r + 1; i < rd; ++i)
        anew[i] = a[i - 1];
      tmp = a[0];
      for (size_t i = 0; i < d; ++i)
        tmp += delta[i] * a[r + i];
      anew[r] = tmp;
    }

    // --- Covariance prediction ---
    if (l > up) {
      if (d == 0) {
        for (size_t i = 0; i < r; ++i) {
          const double vi = (i == 0)       ? 1.0
                            : (i - 1 < q) ? theta[i - 1]
                                           : 0.0;
          for (size_t j = 0; j < r; ++j) {
            tmp = 0.0;
            if (j == 0)
              tmp = vi;
            else if (j - 1 < q)
              tmp = vi * theta[j - 1];
            if (i < p && j < p)
              tmp += phi[i] * phi[j] * P[0];
            if (i < r - 1 && j < r - 1)
              tmp += P[i + 1 + r * (j + 1)];
            if (i < p && j < r - 1)
              tmp += phi[i] * P[j + 1];
            if (j < p && i < r - 1)
              tmp += phi[j] * P[i + 1];
            Pnew[i + r * j] = tmp;
          }
        }
      } else {
        // mm = T @ P
        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            tmp = 0.0;
            if (i < p)
              tmp += phi[i] * P[rd * j];
            if (i < r - 1)
              tmp += P[i + 1 + rd * j];
            mm[i + rd * j] = tmp;
          }
        }
        for (size_t j = 0; j < rd; ++j) {
          tmp = P[rd * j];
          for (size_t k = 0; k < d; ++k)
            tmp += delta[k] * P[r + k + rd * j];
          mm[r + rd * j] = tmp;
        }
        for (size_t i = 1; i < d; ++i) {
          for (size_t j = 0; j < rd; ++j)
            mm[r + i + rd * j] = P[r + i - 1 + rd * j];
        }

        // Pnew = mm @ T'
        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            tmp = 0.0;
            if (i < p)
              tmp += phi[i] * mm[j];
            if (i < r - 1)
              tmp += mm[rd * (i + 1) + j];
            Pnew[j + rd * i] = tmp;
          }
        }
        for (size_t j = 0; j < rd; ++j) {
          tmp = mm[j];
          for (size_t k = 0; k < d; ++k)
            tmp += delta[k] * mm[rd * (r + k) + j];
          Pnew[rd * r + j] = tmp;
        }
        for (size_t i = 1; i < d; ++i) {
          for (size_t j = 0; j < rd; ++j)
            Pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
        }

        // Pnew += V
        for (size_t i = 0; i < q + 1; ++i) {
          const double vi = i == 0 ? 1.0 : theta[i - 1];
          for (size_t j = 0; j < q + 1; ++j) {
            Pnew[i + rd * j] += vi * (j == 0 ? 1.0 : theta[j - 1]);
          }
        }

        // Save mm
        std::copy(mm.begin(), mm.begin() + rd * rd,
                  mm_tape.begin() + l * rd * rd);
      }
    }

    // Save anew, Pnew
    std::copy(anew.begin(), anew.end(), anew_tape.begin() + l * rd);
    std::copy(Pnew.begin(), Pnew.begin() + rd * rd,
              Pnew_tape.begin() + l * rd * rd);

    // --- Innovation and state update ---
    if (!std::isnan(y[l])) {
      observed_tape[l] = 1;
      double resid = y[l] - anew[0];
      for (size_t i = 0; i < d; ++i)
        resid -= delta[i] * anew[r + i];

      for (size_t i = 0; i < rd; ++i) {
        tmp = Pnew[i];
        for (size_t j = 0; j < d; ++j)
          tmp += Pnew[i + (r + j) * rd] * delta[j];
        M[i] = tmp;
      }

      double gain = M[0];
      for (size_t j = 0; j < d; ++j)
        gain += delta[j] * M[r + j];

      resid_tape[l] = resid;
      gain_tape[l] = gain;
      std::copy(M.begin(), M.end(), M_tape.begin() + l * rd);

      if (gain < 1e4) {
        valid_tape[l] = 1;
        nu++;
        if (gain == 0) {
          ssq = std::numeric_limits<double>::infinity();
        } else {
          ssq += resid * resid / gain;
        }
        sumlog += std::log(gain);
      }

      if (gain == 0) {
        for (size_t i = 0; i < rd; ++i) {
          a[i] = std::numeric_limits<double>::infinity();
          for (size_t j = 0; j < rd; ++j)
            Pnew[i + j * rd] = std::numeric_limits<double>::infinity();
        }
      } else {
        for (size_t i = 0; i < rd; ++i) {
          a[i] = anew[i] + M[i] * resid / gain;
          for (size_t j = 0; j < rd; ++j)
            P[i + j * rd] = Pnew[i + j * rd] - M[i] * M[j] / gain;
        }
      }
    } else {
      std::copy(anew.begin(), anew.end(), a.begin());
      std::copy(Pnew.begin(), Pnew.begin() + rd * rd, P.begin());
    }
  }

  // ===================== Compute loss =====================
  if (nu == 0 || ssq <= 0.0 || std::isinf(ssq)) {
    py::array_t<double> d_phiv(p), d_thetav(q), d_Pnv(rd * rd), d_yv(n);
    std::fill(make_span(d_phiv).begin(), make_span(d_phiv).end(), 0.0);
    std::fill(make_span(d_thetav).begin(), make_span(d_thetav).end(), 0.0);
    std::fill(make_span(d_Pnv).begin(), make_span(d_Pnv).end(), 0.0);
    std::fill(make_span(d_yv).begin(), make_span(d_yv).end(), 0.0);
    double loss = std::isinf(ssq) ? std::numeric_limits<double>::max()
                                  : std::numeric_limits<double>::quiet_NaN();
    return {loss, d_phiv, d_thetav, d_Pnv, d_yv};
  }

  const double s2 = ssq / nu;
  const double loss = 0.5 * (std::log(s2) + sumlog / nu);
  const double bar_ssq = 0.5 / ssq;     // dL/d(ssq)
  const double bar_sumlog = 0.5 / nu;    // dL/d(sumlog)

  // ===================== Backward pass =====================
  py::array_t<double> d_phiv(p);
  py::array_t<double> d_thetav(q);
  py::array_t<double> d_Pnv(rd * rd);
  py::array_t<double> d_yv(n);
  auto d_phi = make_span(d_phiv);
  auto d_theta = make_span(d_thetav);
  auto d_Pn = make_span(d_Pnv);
  auto d_y = make_span(d_yv);

  std::fill(d_phi.begin(), d_phi.end(), 0.0);
  std::fill(d_theta.begin(), d_theta.end(), 0.0);
  std::fill(d_Pn.begin(), d_Pn.end(), 0.0);
  std::fill(d_y.begin(), d_y.end(), 0.0);

  // Adjoints of state a and covariance P (propagated backward through time)
  std::vector<double> bar_a(rd, 0.0);
  std::vector<double> bar_P(rd * rd, 0.0);

  // Temporaries for the backward pass
  std::vector<double> bar_anew(rd);
  std::vector<double> bar_Pnew(rd * rd);
  std::vector<double> bar_M(rd);
  std::vector<double> bar_a_prev(rd);
  std::vector<double> bar_P_prev(rd * rd);
  std::vector<double> bar_mm;
  if (d > 0) {
    bar_mm.resize(rd * rd);
  }

  for (size_t l = n - 1;; --l) {
    // Restore tape
    const double *a_l = &a_tape[l * rd];
    const double *P_l = &P_tape[l * rd * rd];
    const double *M_l = &M_tape[l * rd];

    std::fill(bar_anew.begin(), bar_anew.end(), 0.0);
    std::fill(bar_Pnew.begin(), bar_Pnew.end(), 0.0);
    std::fill(bar_M.begin(), bar_M.end(), 0.0);

    if (observed_tape[l] && gain_tape[l] != 0.0) {
      const double resid_l = resid_tape[l];
      const double gain_l = gain_tape[l];
      const double inv_gain = 1.0 / gain_l;
      const double resid_over_gain = resid_l * inv_gain;

      // --- Reverse state update: a_out[i] = anew[i] + M[i]*resid/gain ---
      for (size_t i = 0; i < rd; ++i) {
        bar_anew[i] += bar_a[i];
        bar_M[i] += bar_a[i] * resid_over_gain;
      }
      double bar_resid = 0.0;
      double bar_gain = 0.0;
      for (size_t i = 0; i < rd; ++i) {
        bar_resid += bar_a[i] * M_l[i] * inv_gain;
        bar_gain -= bar_a[i] * M_l[i] * resid_over_gain * inv_gain;
      }

      // --- Reverse covariance update: P_out[i+j*rd] = Pnew[i+j*rd] -
      // M[i]*M[j]/gain ---
      for (size_t i = 0; i < rd; ++i) {
        for (size_t j = 0; j < rd; ++j) {
          const double bP = bar_P[i + j * rd];
          bar_Pnew[i + j * rd] += bP;
          bar_M[i] -= bP * M_l[j] * inv_gain;
          bar_M[j] -= bP * M_l[i] * inv_gain;
          bar_gain += bP * M_l[i] * M_l[j] * inv_gain * inv_gain;
        }
      }

      // --- Reverse likelihood: ssq += resid^2/gain, sumlog += log(gain) ---
      if (valid_tape[l]) {
        bar_resid += bar_ssq * 2.0 * resid_l * inv_gain;
        bar_gain -= bar_ssq * resid_l * resid_l * inv_gain * inv_gain;
        bar_gain += bar_sumlog * inv_gain;
      }

      // --- Reverse gain: gain = M[0] + sum(delta[j]*M[r+j]) ---
      bar_M[0] += bar_gain;
      for (size_t j = 0; j < d; ++j)
        bar_M[r + j] += bar_gain * delta[j];

      // --- Reverse M: M[i] = Pnew[i,0] + sum(Pnew[i,(r+j)]*delta[j]) ---
      for (size_t i = 0; i < rd; ++i) {
        bar_Pnew[i] += bar_M[i];
        for (size_t j = 0; j < d; ++j)
          bar_Pnew[i + (r + j) * rd] += bar_M[i] * delta[j];
      }

      // --- Reverse resid: resid = y[l] - anew[0] - sum(delta[i]*anew[r+i])
      // ---
      d_y[l] += bar_resid;
      bar_anew[0] -= bar_resid;
      for (size_t i = 0; i < d; ++i)
        bar_anew[r + i] -= bar_resid * delta[i];

    } else if (observed_tape[l]) {
      // gain == 0: state goes to inf, adjoint contributions are undefined
      // Pass through: bar_anew = bar_a, bar_Pnew = bar_P
      for (size_t i = 0; i < rd; ++i)
        bar_anew[i] += bar_a[i];
      for (size_t i = 0; i < rd * rd; ++i)
        bar_Pnew[i] += bar_P[i];
    } else {
      // NaN observation: a = anew (copy), P = Pnew (copy)
      for (size_t i = 0; i < rd; ++i)
        bar_anew[i] += bar_a[i];
      for (size_t i = 0; i < rd * rd; ++i)
        bar_Pnew[i] += bar_P[i];
    }

    // --- Reverse covariance prediction ---
    std::fill(bar_P_prev.begin(), bar_P_prev.end(), 0.0);

    if (l > up) {
      if (d == 0) {
        // Reverse: Pnew[i+r*j] = V[i,j] + phi[i]*phi[j]*P[0]
        //                        + P[i+1+r*(j+1)] + phi[i]*P[j+1] +
        //                        phi[j]*P[i+1]
        for (size_t i = 0; i < r; ++i) {
          const double vi = (i == 0)       ? 1.0
                            : (i - 1 < q) ? theta[i - 1]
                                           : 0.0;
          for (size_t j = 0; j < r; ++j) {
            const double b = bar_Pnew[i + r * j];
            if (b == 0.0)
              continue;
            const double vj = (j == 0)       ? 1.0
                              : (j - 1 < q) ? theta[j - 1]
                                             : 0.0;

            // V contribution
            if (i > 0 && i - 1 < q)
              d_theta[i - 1] += b * vj;
            if (j > 0 && j - 1 < q)
              d_theta[j - 1] += b * vi;

            if (i < p && j < p) {
              d_phi[i] += b * phi[j] * P_l[0];
              d_phi[j] += b * phi[i] * P_l[0];
              bar_P_prev[0] += b * phi[i] * phi[j];
            }
            if (i < r - 1 && j < r - 1)
              bar_P_prev[i + 1 + r * (j + 1)] += b;
            if (i < p && j < r - 1) {
              d_phi[i] += b * P_l[j + 1];
              bar_P_prev[j + 1] += b * phi[i];
            }
            if (j < p && i < r - 1) {
              d_phi[j] += b * P_l[i + 1];
              bar_P_prev[i + 1] += b * phi[j];
            }
          }
        }
      } else {
        // d > 0: reverse through mm-based computation
        const double *mm_l = &mm_tape[l * rd * rd];

        // Reverse V addition: Pnew[i+rd*j] += vi*vj
        for (size_t i = 0; i < q + 1; ++i) {
          const double vi = i == 0 ? 1.0 : theta[i - 1];
          for (size_t j = 0; j < q + 1; ++j) {
            const double b = bar_Pnew[i + rd * j];
            if (b == 0.0)
              continue;
            if (i > 0 && i - 1 < q)
              d_theta[i - 1] +=
                  b * (j == 0 ? 1.0 : theta[j - 1]);
            if (j > 0 && j - 1 < q)
              d_theta[j - 1] += b * vi;
          }
        }

        // Reverse Pnew = mm @ T' (stages 6, 5, 4)
        std::fill(bar_mm.begin(), bar_mm.end(), 0.0);

        // Stage 6 reverse: Pnew[rd*(r+i)+j] = mm[rd*(r+i-1)+j]
        for (size_t i = d - 1; i >= 1; --i) {
          for (size_t j = 0; j < rd; ++j)
            bar_mm[rd * (r + i - 1) + j] += bar_Pnew[rd * (r + i) + j];
        }

        // Stage 5 reverse: Pnew[rd*r+j] = mm[j] +
        // sum(delta[k]*mm[rd*(r+k)+j])
        for (size_t j = 0; j < rd; ++j) {
          const double b = bar_Pnew[rd * r + j];
          bar_mm[j] += b;
          for (size_t k = 0; k < d; ++k)
            bar_mm[rd * (r + k) + j] += b * delta[k];
        }

        // Stage 4 reverse: Pnew[j+rd*i] = phi[i]*mm[j] + mm[rd*(i+1)+j]
        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            const double b = bar_Pnew[j + rd * i];
            if (i < p) {
              d_phi[i] += b * mm_l[j];
              bar_mm[j] += b * phi[i];
            }
            if (i < r - 1)
              bar_mm[rd * (i + 1) + j] += b;
          }
        }

        // Reverse mm = T @ P (stages 3, 2, 1)
        // Stage 3 reverse: mm[r+i+rd*j] = P[r+i-1+rd*j]
        for (size_t i = d - 1; i >= 1; --i) {
          for (size_t j = 0; j < rd; ++j)
            bar_P_prev[r + i - 1 + rd * j] += bar_mm[r + i + rd * j];
        }

        // Stage 2 reverse: mm[r+rd*j] = P[rd*j] +
        // sum(delta[k]*P[r+k+rd*j])
        for (size_t j = 0; j < rd; ++j) {
          const double b = bar_mm[r + rd * j];
          bar_P_prev[rd * j] += b;
          for (size_t k = 0; k < d; ++k)
            bar_P_prev[r + k + rd * j] += b * delta[k];
        }

        // Stage 1 reverse: mm[i+rd*j] = phi[i]*P[rd*j] + P[i+1+rd*j]
        for (size_t i = 0; i < r; ++i) {
          for (size_t j = 0; j < rd; ++j) {
            const double b = bar_mm[i + rd * j];
            if (i < p) {
              d_phi[i] += b * P_l[rd * j];
              bar_P_prev[rd * j] += b * phi[i];
            }
            if (i < r - 1)
              bar_P_prev[i + 1 + rd * j] += b;
          }
        }
      }
    } else {
      // l <= up: Pnew was the initial Pn, accumulate into d_Pn
      for (size_t i = 0; i < rd * rd; ++i)
        d_Pn[i] += bar_Pnew[i];
    }

    // --- Reverse state prediction ---
    std::fill(bar_a_prev.begin(), bar_a_prev.end(), 0.0);
    for (size_t i = 0; i < r; ++i) {
      if (i < r - 1)
        bar_a_prev[i + 1] += bar_anew[i];
      if (i < p) {
        d_phi[i] += bar_anew[i] * a_l[0];
        bar_a_prev[0] += bar_anew[i] * phi[i];
      }
    }
    if (d > 0) {
      // anew[r] = a[0] + sum(delta[i]*a[r+i])
      bar_a_prev[0] += bar_anew[r];
      for (size_t i = 0; i < d; ++i)
        bar_a_prev[r + i] += bar_anew[r] * delta[i];
      // anew[r+i] = a[r+i-1] for i=1..d-1
      for (size_t i = 1; i < d; ++i)
        bar_a_prev[r + i - 1] += bar_anew[r + i];
    }

    // Propagate to next backward step
    std::copy(bar_a_prev.begin(), bar_a_prev.end(), bar_a.begin());
    std::copy(bar_P_prev.begin(), bar_P_prev.end(), bar_P.begin());

    if (l == 0)
      break;
  }

  return {loss, d_phiv, d_thetav, d_Pnv, d_yv};
}

void inclu2(const std::vector<double> &xnext, std::vector<double> &xrow,
            double ynext, std::span<double> d, std::vector<double> &rbar,
            std::vector<double> &thetab) {
  const size_t np = xnext.size();
  assert(xrow.size() == np);
  assert(thetab.size() == np);

  std::copy(xnext.begin(), xnext.end(), xrow.begin());
  size_t ithisr = 0;

  for (size_t i = 0; i < np; ++i) {
    if (xrow[i] != 0.0) {
      const double xi = xrow[i];
      const double di = d[i];
      const double dpi = di + xi * xi;
      d[i] = dpi;

      const auto [cbar, sbar] = [dpi, di, xi]() -> std::pair<double, double> {
        if (dpi == 0) {
          return {std::numeric_limits<double>::infinity(),
                  std::numeric_limits<double>::infinity()};
        } else {
          return {di / dpi, xi / dpi};
        }
      }();

      for (size_t k = i + 1; k < np; ++k) {
        const double xk = xrow[k];
        const double rbthis = rbar[ithisr];
        xrow[k] = xk - xi * rbthis;
        rbar[ithisr++] = cbar * rbthis + sbar * xk;
      }

      const double xk = ynext;
      ynext = xk - xi * thetab[i];
      thetab[i] = cbar * thetab[i] + sbar * xk;

      if (di == 0.0) {
        return;
      }
    } else {
      ithisr += np - i - 1;
    }
  }
}

void getQ0(const py::array_t<double> phiv, const py::array_t<double> thetav,
           py::array_t<double> resv) {
  assert(thetav.ndim() == 1);
  assert(phiv.ndim() == 1);
  assert(resv.ndim() == 1);

  const auto phi = make_cspan(phiv);
  const auto theta = make_cspan(thetav);
  const auto res = make_span(resv);

  const size_t p = phi.size();
  const size_t q = theta.size();
  const size_t r = std::max(p, q + 1);
  const size_t np = r * (r + 1) / 2;
  const size_t nrbar = np * (np - 1) / 2;

  std::vector<double> V(np);

  {
    size_t ind = 0;

    for (size_t j = 0; j < r; ++j) {
      double vj = 0.0;
      if (j == 0) {
        vj = 1.0;
      } else if (j - 1 < q) {
        vj = theta[j - 1];
      }

      for (size_t i = j; i < r; ++i) {
        double vi = 0.0;
        if (i == 0) {
          vi = 1.0;
        } else if (i - 1 < q) {
          vi = theta[i - 1];
        }
        V[ind++] = vi * vj;
      }
    }
  }

  if (r == 1) {
    if (p == 0) {
      res[0] = 1.0;
    } else {
      res[0] = 1.0 / (1 - phi[0] * phi[0]);
    }
    return;
  }

  if (p > 0) {
    std::vector<double> rbar(nrbar);
    std::vector<double> thetab(np);
    std::vector<double> xnext(np);
    std::vector<double> xrow(np);

    size_t ind = 0;
    py::ssize_t ind1 = -1;

    const size_t npr = np - r;
    const size_t npr1 = npr + 1;
    size_t indj = npr;
    size_t ind2 = npr - 1;

    for (size_t j = 0; j < r; ++j) {
      const double phij = j < p ? phi[j] : 0.0;
      size_t indi = npr1 + j;

      xnext[indj++] = 0.0;

      for (size_t i = j; i < r; ++i) {
        const double ynext = V[ind++];
        const double phii = i < p ? phi[i] : 0.0;

        if (j != r - 1) {
          xnext[indj] = -phii;
          if (i != r - 1) {
            xnext[indi] -= phij;
            xnext[++ind1] = -1.0;
          }
        }

        xnext[npr] = -phii * phij;

        if (++ind2 >= np) {
          ind2 = 0;
        }

        xnext[ind2] += 1.0;
        inclu2(xnext, xrow, ynext, res, rbar, thetab);
        xnext[ind2] = 0.0;

        if (i != r - 1) {
          xnext[indi++] = 0.0;
          xnext[ind1] = 0.0;
        }
      }
    }

    size_t ithisr = nrbar - 1;
    size_t im = np - 1;
    for (size_t i = 0; i < np; ++i) {
      size_t jm = np - 1;
      double bi = thetab[im];

      for (size_t j = 0; j < i; ++j) {
        bi -= rbar[ithisr--] * res[jm--];
      }

      res[im--] = bi;
    }

    ind = npr;

    for (size_t i = 0; i < r; ++i) {
      xnext[i] = res[ind++];
    }

    ind = np - 1;
    ind1 = npr - 1;
    for (size_t i = 0; i < npr; ++i) {
      res[ind--] = res[ind1--];
    }
    std::copy(xnext.begin(), xnext.begin() + r, res.begin());
  } else {
    size_t ind = np;
    size_t indn = np;
    for (size_t i = 0; i < r; ++i) {
      for (size_t j = 0; j < i + 1; ++j) {
        --ind;
        res[ind] = V[ind];
        if (j != 0) {
          res[ind] += res[--indn];
        }
      }
    }
  }

  {
    size_t ind = np;

    for (size_t i = r - 1; i > 0; --i) {
      for (size_t j = r - 1; j > i - 1; --j) {
        res[r * i + j] = res[--ind];
      }
    }

    for (size_t i = 0; i < r - 1; ++i) {
      for (size_t j = i + 1; j < r; ++j) {
        res[i + r * j] = res[j + r * i];
      }
    }
  }
}

py::array_t<double> arima_gradtrans(const py::array_t<double> xv,
                                    const py::array_t<int32_t> armav) {
  assert(xv.ndim() == 1);
  assert(armav.ndim() == 1);

  constexpr double eps = 1e-3;
  const auto arma = make_cspan(armav);
  const auto x = make_cspan(xv);
  const size_t n = x.size();

  assert(arma.size() == 7);
  assert(arma[0] >= 0 && arma[1] >= 0 && arma[2] >= 0 && arma[3] >= 0 &&
         arma[4] >= 0 && arma[5] >= 0 && arma[6] >= 0);

  const auto mp = static_cast<size_t>(arma[0]);
  const auto mq = static_cast<size_t>(arma[1]);
  const auto msp = static_cast<size_t>(arma[2]);

  std::array<double, 100> w1;
  std::array<double, 100> w2;
  std::array<double, 100> w3;
  assert(mp < 100);

  py::array_t<double> outv({n, n});
  const auto out = make_span(outv);
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      out[i * n + j] = (i == j) ? 1.0 : 0.0;
    }
  }

  if (mp > 0) {
    std::copy(x.begin(), x.begin() + mp, w1.begin());
    partrans(mp, w1, w2);

    for (size_t i = 0; i < mp; ++i) {
      w1[i] += eps;
      partrans(mp, w1, w3);

      for (size_t j = 0; j < mp; ++j) {
        out[n * i + j] = (w3[j] - w2[j]) / eps;
      }
      w1[i] -= eps;
    }
  }

  if (msp > 0) {
    const size_t v = mp + mq;
    std::copy(x.begin() + v, x.begin() + v + msp, w1.begin());
    partrans(msp, w1, w2);

    for (size_t i = 0; i < msp; ++i) {
      w1[i] += eps;
      partrans(msp, w1, w3);

      for (size_t j = 0; j < msp; ++j) {
        out[n * (i + v) + j + v] = (w3[j] - w2[j]) / eps;
      }
      w1[i] -= eps;
    }
  }

  return outv;
}

py::array_t<double> arima_undopars(const py::array_t<double> xv,
                                   const py::array_t<int32_t> armav) {
  assert(xv.ndim() == 1);
  assert(armav.ndim() == 1);

  const auto x = make_cspan(xv);
  const auto arma = make_cspan(armav);

  assert(arma.size() == 7);
  assert(arma[0] >= 0 && arma[1] >= 0 && arma[2] >= 0 && arma[3] >= 0 &&
         arma[4] >= 0 && arma[5] >= 0 && arma[6] >= 0);

  const int32_t mp = arma[0];
  const int32_t mq = arma[1];
  const int32_t msp = arma[2];

  py::array_t<double> outv(xv.size());
  const auto out = make_span(outv);

  std::copy(x.begin(), x.end(), out.begin());

  if (mp > 0) {
    partrans(mp, x, out);
  }

  const size_t v = mp + mq;

  if (msp > 0) {
    partrans(msp, x.subspan(v), out.subspan(v));
  }

  return outv;
}

void invpartrans(const uint32_t p, const py::array_t<double> phiv,
                 py::array_t<double> outv) {
  assert(phiv.ndim() == 1);
  assert(outv.ndim() == 1);
  assert(p <= phiv.size());
  assert(p <= outv.size());

  const auto phi = make_cspan(phiv);
  const auto out = make_span(outv);

  std::copy(phi.begin(), phi.begin() + p, out.begin());

  std::vector<double> work(phi.begin(), phi.begin() + p);
  for (size_t j = p - 1; j > 0; --j) {
    const double a = out[j];
    for (size_t k = 0; k < j; ++k) {
      work[k] = (out[k] + a * out[j - k - 1]) / (1 - a * a);
    }
    std::copy(work.begin(), work.begin() + j, out.begin());
  }

  for (size_t j = 0; j < p; ++j) {
    out[j] = std::atanh(out[j]);
  }
}

void init(py::module_ &m) {
  py::module_ arima = m.def_submodule("arima");
  arima.def("arima_css", &arima_css);
  arima.def("arima_css_grad", &arima_css_grad);
  arima.def("arima_like", &arima_like);
  arima.def("arima_like_grad", &arima_like_grad);
  arima.def("getQ0", &getQ0);
  arima.def("arima_gradtrans", &arima_gradtrans);
  arima.def("arima_undopars", &arima_undopars);
  arima.def("invpartrans", &invpartrans);
  arima.def("arima_transpar", &arima_transpar);
}
} // namespace arima
