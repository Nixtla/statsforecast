#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "arima.h"

void partrans(int p, const double *raw, double *newv) {
  std::transform(raw, raw + p, newv, [](double x) { return std::tanh(x); });
  std::vector<double> work(newv, newv + p);
  for (int j = 1; j < p; ++j) {
    for (int k = 0; k < j; ++k) {
      work[k] -= newv[j] * newv[j - k - 1];
    }
    std::copy(work.begin(), work.begin() + j, newv);
  }
}

struct Trarma {
  std::vector<double> phi;
  std::vector<double> theta;
};

Trarma arima_transpar(const double *params_in, const int *arma, bool trans) {
  int mp = arma[0], mq = arma[1], msp = arma[2], msq = arma[3], ns = arma[4];
  int p = mp + ns * msp;
  int q = mq + ns * msq;
  std::vector<double> phi(p, 0.0);
  std::vector<double> theta(q, 0.0);
  std::vector<double> params(params_in, params_in + mp + mq + msp + msq);
  if (trans) {
    if (mp > 0) {
      partrans(mp, params_in, params.data());
    }
    int v = mp + mq;
    if (msp > 0) {
      partrans(msp, params_in + v, params.data() + v);
    }
  }
  if (ns > 0) {
    std::copy(params.begin(), params.begin() + mp, phi.begin());
    std::fill(phi.begin() + mp, phi.begin() + p, 0.0);
    std::copy(params.begin() + mp, params.begin() + mp + mq, theta.begin());
    std::fill(theta.begin() + mq, theta.begin() + q, 0.0);
    for (int j = 0; j < msp; ++j) {
      phi[(j + 1) * ns - 1] += params[j + mp + mq];
      for (int i = 0; i < mp; ++i) {
        phi[(j + 1) * ns + i] -= params[i] * params[j + mp + mq];
      }
    }
    for (int j = 0; j < msq; ++j) {
      theta[(j + 1) * ns - 1] += params[j + mp + mq + msp];
      for (int i = 0; i < mq; ++i) {
        theta[(j + 1) * ns + i] += params[i + mp] * params[j + mp + mq + msp];
      }
    }
  } else {
    std::copy(params.begin(), params.begin() + mp, phi.begin());
    std::copy(params.begin() + mp, params.begin() + mp + mq, theta.begin());
  }
  return {phi, theta};
}

double arima_css(const double *y, int n, const int *arma, const double *phi,
                 int p, const double *theta, int q) {
  int nu = 0;
  double ssq = 0.0;
  int ncond = arma[0] + arma[5] + arma[4] * (arma[2] + arma[6]);
  std::vector<double> w(y, y + n);
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
  std::vector<double> resid(n);

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
  return ssq / nu;
}

void PrintVector(const std::string &name, const std::vector<double> &x) {
  std::cout << name << ": " << std::fixed;
  for (const auto v : x) {
    std::cout << std::setprecision(3) << v << " ";
  }
  std::cout << std::endl;
}

double arma_css_op(const double *p, const double *y, int n, const double *coef,
                   const int *arma, const bool *mask) {
  int narma = arma[0] + arma[1] + arma[2] + arma[3];
  std::vector<double> par(coef, coef + narma);
  for (int i = 0; i < narma; ++i) {
    if (mask[i]) {
      par[i] = p[i];
    }
  }
  Trarma trarma = arima_transpar(par.data(), arma, false);
#ifdef DEBUG
  PrintVector("phi", trarma.phi);
  PrintVector("theta", trarma.theta);
#endif
  double res = arima_css(y, n, arma, trarma.phi.data(), trarma.phi.size(),
                         trarma.theta.data(), trarma.theta.size());
  if (!std::isfinite(res)) {
    return std::numeric_limits<double>::max();
  }
  if (res <= 0) {
    return -std::numeric_limits<double>::infinity();
  }
#ifdef DEBUG
  PrintVector("par", par);
  std::cout << "res: " << 0.5 * std::log(res) << std::endl;
#endif
  return 0.5 * std::log(res);
}

void arima_like(const double *y, int n, const double *phi, int p,
                const double *theta, int q, const double *delta, int d,
                double *a, int rd, double *P, double *Pnew, int up,
                bool use_resid, double *ssq, double *sumlog, int *nu,
                double *rsResid) {
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
        (*nu)++;
        if (gain == 0) {
          *ssq = std::numeric_limits<double>::infinity();
        } else {
          *ssq += resid * resid / gain;
        }
        *sumlog += std::log(gain);
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
    }
  }
}

void inclu2(int np, const double *xnext, double *xrow, double ynext, double *d,
            double *rbar, double *thetab) {
  std::copy(xnext, xnext + np, xrow);
  int ithisr = 0;
  for (int i = 0; i < np; ++i) {
    if (xrow[i] != 0.0) {
      double xi = xrow[i];
      double di = d[i];
      double dpi = di + xi * xi;
      d[i] = dpi;
      double cbar, sbar;
      if (dpi == 0) {
        cbar = std::numeric_limits<double>::infinity();
        sbar = std::numeric_limits<double>::infinity();
      } else {
        cbar = di / dpi;
        sbar = xi / dpi;
      }
      for (int k = i + 1; k < np; ++k) {
        double xk = xrow[k];
        double rbthis = rbar[ithisr];
        xrow[k] = xk - xi * rbthis;
        rbar[ithisr++] = cbar * rbthis + sbar * xk;
      }
      double xk = ynext;
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

void getQ0(const double *phi, int p, const double *theta, int q, double *res) {
  int r = std::max(p, q + 1);
  int np = r * (r + 1) / 2;
  int nrbar = np * (np - 1) / 2;
  std::vector<double> V(np);
  int ind = 0;
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
        inclu2(np, xnext.data(), xrow.data(), ynext, res, rbar.data(),
               thetab.data());
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

void upARIMA(const double *phi, int p, const double *theta, int q, int d,
             double *Pn, double *T, double *a) {
  int r = std::max(p, q + 1);
  int rd = r + d;
  if (p > 0) {
    for (int i = 0; i < p; ++i) {
      T[i * rd] = phi[i];
    }
  }
  if (r > 1) {
    auto res = new double[r * r]();
    getQ0(phi, p, theta, q, res);
    for (int i = 0; i < r; ++i) {
      std::copy(res + i * r, res + (i + 1) * r, Pn + i * rd);
    }
    delete[] res;
  } else {
    Pn[0] = 1.0;
    if (p > 0) {
      Pn[0] /= (1 - phi[0] * phi[0]);
    }
  }
  std::fill(a, a + rd, 0.0);
}

void arima_gradtrans(const double *x, int n, const int *arma, double *out) {
  double eps = 1e-3;
  int mp = arma[0], mq = arma[1], msp = arma[2];
  double *w1 = new double[100];
  double *w2 = new double[100];
  double *w3 = new double[100];
  if (mp > 0) {
    std::copy(x, x + mp, w1);
    partrans(mp, w1, w2);
    for (int i = 0; i < mp; ++i) {
      w1[i] += eps;
      partrans(mp, w1, w3);
      for (int j = 0; j < mp; ++j) {
        out[i * n + j] = (w3[j] - w2[j]) / eps;
      }
      w1[i] -= eps;
    }
  }
  if (msp > 0) {
    int v = mp + mq;
    std::copy(x + v, x + v + msp, w1);
    partrans(msp, w1, w2);
    for (int i = 0; i < msp; ++i) {
      w1[i] += eps;
      partrans(msp, w1, w3);
      for (int j = 0; j < msp; ++j) {
        out[(i + v) * n + v + j] = (w3[j] - w2[j]) / eps;
      }
      w1[1] -= eps;
    }
  }
  delete[] w1;
  delete[] w2;
  delete[] w3;
}

double armafn(const double *p, const double *y, int n, const double *delta,
              int d, const double *coef, const int *arma, const bool *mask,
              bool trans, double *P, double *Pn, double *a, double *T) {
  int narma = arma[0] + arma[1] + arma[2] + arma[3];
  std::vector<double> par(coef, coef + narma);
  for (int i = 0; i < narma; ++i) {
    if (mask[i]) {
      par[i] = p[i];
    }
  }
  Trarma trarma = arima_transpar(par.data(), arma, trans);
  upARIMA(trarma.phi.data(), trarma.phi.size(), trarma.theta.data(),
          trarma.theta.size(), d, Pn, T, a);
  int r = std::max(trarma.phi.size(), trarma.theta.size() + 1);
  int rd = r + d;
  double rsResid;
  double ssq = 0.0;
  double sumlog = 0.0;
  int nu = 0;
  arima_like(y, n, trarma.phi.data(), trarma.phi.size(), trarma.theta.data(),
             trarma.theta.size(), delta, d, a, rd, P, Pn, 0, false, &ssq,
             &sumlog, &nu, &rsResid);
  if (nu == 0) {
    return std::numeric_limits<double>::infinity();
  }
  double s2 = ssq / nu;
  if (s2 <= 0) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return 0.5 * (std::log(s2) + sumlog / nu);
}

void arima_undopars(const double *x, const int *arma, double *out) {
  int mp = arma[0], mq = arma[1], msp = arma[2];
  if (mp > 0) {
    partrans(mp, x, out);
  }
  int v = mp + mq;
  if (msp > 0) {
    partrans(msp, x + v, out + v);
  }
}

void tsconv(const double *a, int na, const double *b, int nb, double *out) {
  for (int i = 0; i < na; ++i) {
    for (int j = 0; j < nb; ++j) {
      out[i + j] += a[i] * b[j];
    }
  }
}

void invpartrans(int p, const double *phi, double *out) {
  std::copy(phi, phi + p, out);
  std::vector<double> work(phi, phi + p);
  for (int j = p - 1; j > 0; --j) {
    double a = out[j];
    for (int k = 0; k < j; ++k) {
      work[k] = (out[k] + a * out[j - k - 1]) / (1 - a * a);
      out[k] = work[k];
    }
  }
  for (int j = 0; j < p; ++j) {
    out[j] = std::atanh(out[j]);
  }
}
