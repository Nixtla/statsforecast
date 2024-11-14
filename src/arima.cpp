#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <span>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace arima {
namespace py = pybind11;
namespace {
template <typename T> std::span<T> make_span(py::array_t<T> &array) {
  // ssize_t to size_t is safe because the size of an array is non-negative
  return {array.mutable_data(), static_cast<size_t>(array.size())};
}

template <typename T>
std::span<const T> make_cspan(const py::array_t<T> &array) {
  // ssize_t to size_t is safe because the size of an array is non-negative
  return {array.data(), static_cast<size_t>(array.size())};
}
} // namespace

void partrans(const size_t p, const std::span<const double> rawv,
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
               const py::array_t<int> armav, bool trans) {
  assert(params_inv.ndim() == 1);
  assert(armav.ndim() == 1);

  const auto arma = make_cspan(armav);
  const auto params_in = make_cspan(params_inv);

  const int mp = arma[0];
  const int mq = arma[1];
  const int msp = arma[2];
  const int msq = arma[3];
  const int ns = arma[4];
  assert(mp >= 0);
  assert(mq >= 0);
  assert(msp >= 0);
  assert(msq >= 0);
  assert(ns >= 0);
  const int p = mp + ns * msp;
  const int q = mq + ns * msq;

  std::vector<double> params(params_in.begin(), params_in.end());

  py::array_t<double> phiv(p);
  py::array_t<double> thetav(q);
  const auto phi = make_span(phiv);
  const auto theta = make_span(thetav);

  if (trans) {
    if (mp > 0) {
      partrans(mp, params_in, params);
    }
    int v = mp + mq;
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
arima_css(const py::array_t<double> yv, const py::array_t<int> armav,
          const py::array_t<double> phiv, const py::array_t<double> thetav) {
  int n = static_cast<int>(yv.size());
  int p = static_cast<int>(phiv.size());
  int q = static_cast<int>(thetav.size());
  auto y = yv.data();
  auto arma = armav.data();
  auto phi = phiv.data();
  auto theta = thetav.data();
  int ncond = arma[0] + arma[5] + arma[4] * (arma[2] + arma[6]);
  int nu = 0;
  double ssq = 0.0;

  auto residv = py::array_t<double>(n);
  auto resid = residv.mutable_data();
  auto w = std::vector<double>(y, y + yv.size());
  for (int _ = 0; _ < arma[5]; ++_) {
    for (int l = n - 1; l > 0; --l) {
      w[l] -= w[l - 1];
    }
  }
  int ns = arma[4];
  for (int _ = 0; _ < arma[6]; ++_) {
    for (int l = n - 1; l >= ns; --l) {
      w[l] -= w[l - ns];
    }
  }
  for (int l = ncond; l < n; ++l) {
    double tmp = w[l];
    for (int j = 0; j < p; ++j) {
      tmp -= phi[j] * w[l - j - 1];
    }
    for (int j = 0; j < std::min(l - ncond, q); ++j) {
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

std::tuple<double, double, int>
arima_like(const py::array_t<double> yv, const py::array_t<double> phiv,
           const py::array_t<double> thetav, const py::array_t<double> deltav,
           py::array_t<double> av, py::array_t<double> Pv,
           py::array_t<double> Pnewv, int up, bool use_resid,
           py::array_t<double> rsResidv) {
  int n = static_cast<int>(yv.size());
  int d = static_cast<int>(deltav.size());
  int rd = static_cast<int>(av.size());
  int p = static_cast<int>(phiv.size());
  int q = static_cast<int>(thetav.size());
  auto y = yv.data();
  auto phi = phiv.data();
  auto theta = thetav.data();
  auto delta = deltav.data();
  auto a = av.mutable_data();
  auto P = Pv.mutable_data();
  auto Pnew = Pnewv.mutable_data();
  auto rsResid = rsResidv.mutable_data();
  double ssq = 0.0;
  double sumlog = 0.0;
  int nu = 0;
  int r = rd - d;

  std::vector<double> anew(rd);
  std::vector<double> M(rd);
  std::vector<double> mm;
  if (d > 0) {
    mm.resize(rd * rd);
  }
  double tmp;
  for (int l = 0; l < n; ++l) {
    for (int i = 0; i < r; ++i) {
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
      for (int i = r + 1; i < rd; ++i) {
        anew[i] = a[i - 1];
      }
      tmp = a[0];
      for (int i = 0; i < d; ++i) {
        tmp += delta[i] * a[r + i];
      }
      anew[r] = tmp;
    }
    if (l > up) {
      if (d == 0) {
        for (int i = 0; i < r; ++i) {
          double vi = 0.0;
          if (i == 0) {
            vi = 1.0;
          } else if (i - 1 < q) {
            vi = theta[i - 1];
          }
          for (int j = 0; j < r; ++j) {
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
        for (int i = 0; i < r; ++i) {
          for (int j = 0; j < rd; ++j) {
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
        for (int j = 0; j < rd; ++j) {
          tmp = P[rd * j];
          for (int k = 0; k < d; ++k) {
            tmp += delta[k] * P[r + k + rd * j];
          }
          mm[r + rd * j] = tmp;
        }
        for (int i = 1; i < d; ++i) {
          for (int j = 0; j < rd; ++j) {
            mm[r + i + rd * j] = P[r + i - 1 + rd * j];
          }
        }
        for (int i = 0; i < r; ++i) {
          for (int j = 0; j < rd; ++j) {
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
        for (int j = 0; j < rd; ++j) {
          tmp = mm[j];
          for (int k = 0; k < d; ++k) {
            tmp += delta[k] * mm[rd * (r + k) + j];
          }
          Pnew[rd * r + j] = tmp;
        }
        for (int i = 1; i < d; ++i) {
          for (int j = 0; j < rd; ++j) {
            Pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
          }
        }
        for (int i = 0; i < q + 1; ++i) {
          double vi;
          if (i == 0) {
            vi = 1.0;
          } else {
            vi = theta[i - 1];
          }
          for (int j = 0; j < q + 1; ++j) {
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
      for (int i = 0; i < d; ++i) {
        resid -= delta[i] * anew[r + i];
      }
      for (int i = 0; i < rd; ++i) {
        tmp = Pnew[i];
        for (int j = 0; j < d; ++j) {
          tmp += Pnew[i + (r + j) * rd] * delta[j];
        }
        M[i] = tmp;
      }
      double gain = M[0];
      for (int j = 0; j < d; ++j) {
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
        for (int i = 0; i < rd; ++i) {
          a[i] = std::numeric_limits<double>::infinity();
          for (int j = 0; j < rd; ++j) {
            Pnew[i + j * rd] = std::numeric_limits<double>::infinity();
          }
        }
      } else {
        for (int i = 0; i < rd; ++i) {
          a[i] = anew[i] + M[i] * resid / gain;
          for (int j = 0; j < rd; ++j) {
            P[i + j * rd] = Pnew[i + j * rd] - M[i] * M[j] / gain;
          }
        }
      }
    } else {
      std::copy(anew.begin(), anew.end(), a);
      std::copy(Pnew, Pnew + rd * rd, P);
      if (use_resid) {
        rsResid[l] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  return {ssq, sumlog, nu};
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
  auto phi = phiv.data();
  auto theta = thetav.data();
  auto res = resv.mutable_data();
  int p = static_cast<int>(phiv.size());
  int q = static_cast<int>(thetav.size());
  int r = std::max(p, q + 1);
  int np = r * (r + 1) / 2;
  int nrbar = np * (np - 1) / 2;
  int ind = 0;

  std::vector<double> V(np);
  for (int j = 0; j < r; ++j) {
    double vj = 0.0;
    if (j == 0) {
      vj = 1.0;
    } else if (j - 1 < q) {
      vj = theta[j - 1];
    }
    for (int i = j; i < r; ++i) {
      double vi = 0.0;
      if (i == 0) {
        vi = 1.0;
      } else if (i - 1 < q) {
        vi = theta[i - 1];
      }
      V[ind++] = vi * vj;
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
    ind = 0;
    int ind1 = -1;
    int npr = np - r;
    int npr1 = npr + 1;
    int indj = npr;
    int ind2 = npr - 1;
    for (int j = 0; j < r; ++j) {
      double phij = j < p ? phi[j] : 0.0;
      xnext[indj++] = 0.0;
      int indi = npr1 + j;
      for (int i = j; i < r; ++i) {
        double ynext = V[ind++];
        double phii = i < p ? phi[i] : 0.0;
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
        inclu2(xnext, xrow, ynext, std::span(res, resv.size()), rbar, thetab);
        xnext[ind2] = 0.0;
        if (i != r - 1) {
          xnext[indi++] = 0.0;
          xnext[ind1] = 0.0;
        }
      }
    }
    int ithisr = nrbar - 1;
    int im = np - 1;
    for (int i = 0; i < np; ++i) {
      double bi = thetab[im];
      int jm = np - 1;
      for (int j = 0; j < i; ++j) {
        bi -= rbar[ithisr--] * res[jm--];
      }
      res[im--] = bi;
    }
    ind = npr;
    for (int i = 0; i < r; ++i) {
      xnext[i] = res[ind++];
    }
    ind = np - 1;
    ind1 = npr - 1;
    for (int i = 0; i < npr; ++i) {
      res[ind--] = res[ind1--];
    }
    std::copy(xnext.begin(), xnext.begin() + r, res);
  } else {
    int indn = np;
    ind = np;
    for (int i = 0; i < r; ++i) {
      for (int j = 0; j < i + 1; ++j) {
        --ind;
        res[ind] = V[ind];
        if (j != 0) {
          res[ind] += res[--indn];
        }
      }
    }
  }
  ind = np;
  for (int i = r - 1; i > 0; --i) {
    for (int j = r - 1; j > i - 1; --j) {
      res[r * i + j] = res[--ind];
    }
  }
  for (int i = 0; i < r - 1; ++i) {
    for (int j = i + 1; j < r; ++j) {
      res[i + r * j] = res[j + r * i];
    }
  }
}

py::array_t<double> arima_gradtrans(const py::array_t<double> xv,
                                    const py::array_t<int> armav) {
  assert(xv.ndim() == 1);
  assert(armav.ndim() == 1);

  constexpr double eps = 1e-3;
  const auto arma = make_cspan(armav);
  const auto x = make_cspan(xv);
  const size_t n = x.size();

  const int mp = arma[0];
  const int mq = arma[1];
  const int msp = arma[2];

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
                                   const py::array_t<int> armav) {
  assert(xv.ndim() == 1);
  assert(armav.ndim() == 1);

  const auto x = make_cspan(xv);
  const auto arma = make_cspan(armav);

  const int mp = arma[0];
  const int mq = arma[1];
  const int msp = arma[2];
  assert(mp >= 0);
  assert(mq >= 0);
  assert(msp >= 0);

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

void invpartrans(const int p, const py::array_t<double> phiv,
                 py::array_t<double> outv) {
  assert(p >= 0);
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
  arima.def("arima_like", &arima_like);
  arima.def("getQ0", &getQ0);
  arima.def("arima_gradtrans", &arima_gradtrans);
  arima.def("arima_undopars", &arima_undopars);
  arima.def("invpartrans", &invpartrans);
  arima.def("arima_transpar", &arima_transpar);
}
} // namespace arima
