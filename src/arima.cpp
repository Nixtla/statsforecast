#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

namespace arima
{
  namespace py = pybind11;
  using Eigen::VectorXd;
  using Eigen::VectorXi;
  using RowMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  template <typename T>
  using Ref = Eigen::Ref<T>;
  template <typename T>
  using CRef = const Eigen::Ref<const T> &;

  void partrans(int p, const double *raw, double *newv)
  {
    std::transform(raw, raw + p, newv, [](double x)
                   { return std::tanh(x); });
    std::vector<double> work(newv, newv + p);
    for (int j = 1; j < p; ++j)
    {
      for (int k = 0; k < j; ++k)
      {
        work[k] -= newv[j] * newv[j - k - 1];
      }
      std::copy(work.begin(), work.begin() + j, newv);
    }
  }

  std::tuple<VectorXd, VectorXd> arima_transpar(CRef<VectorXd> params_in, CRef<VectorXi> arma, bool trans)
  {
    int mp = arma[0];
    int mq = arma[1];
    int msp = arma[2];
    int msq = arma[3];
    int ns = arma[4];
    int p = mp + ns * msp;
    int q = mq + ns * msq;
    int n = mp + mq + msp + msq;
    auto params = std::vector<double>(n);
    VectorXd phi = VectorXd::Zero(p);
    VectorXd theta = VectorXd::Zero(q);
    std::copy(params_in.begin(), params_in.begin() + n, params.begin());
    if (trans)
    {
      if (mp > 0)
      {
        partrans(mp, params_in.data(), params.data());
      }
      int v = mp + mq;
      if (msp > 0)
      {
        partrans(msp, params_in.data() + v, params.data() + v);
      }
    }
    if (ns > 0)
    {
      std::copy(params.begin(), params.begin() + mp, phi.data());
      std::fill(phi.data() + mp, phi.data() + p, 0.0);
      std::copy(params.begin() + mp, params.begin() + mp + mq, theta.data());
      std::fill(theta.data() + mq, theta.data() + q, 0.0);
      for (int j = 0; j < msp; ++j)
      {
        phi[(j + 1) * ns - 1] += params[j + mp + mq];
        for (int i = 0; i < mp; ++i)
        {
          phi[(j + 1) * ns + i] -= params[i] * params[j + mp + mq];
        }
      }
      for (int j = 0; j < msq; ++j)
      {
        theta[(j + 1) * ns - 1] += params[j + mp + mq + msp];
        for (int i = 0; i < mq; ++i)
        {
          theta[(j + 1) * ns + i] += params[i + mp] * params[j + mp + mq + msp];
        }
      }
    }
    else
    {
      std::copy(params.begin(), params.begin() + mp, phi.data());
      std::copy(params.begin() + mp, params.begin() + mp + mq, theta.data());
    }
    return {phi, theta};
  }

  std::tuple<double, VectorXd> arima_css(CRef<VectorXd> y, CRef<VectorXi> arma, CRef<VectorXd> phi,
                                         CRef<VectorXd> theta)
  {
    int n = static_cast<int>(y.size());
    int p = static_cast<int>(phi.size());
    int q = static_cast<int>(theta.size());
    int ncond = arma[0] + arma[5] + arma[4] * (arma[2] + arma[6]);
    int nu = 0;
    double ssq = 0.0;

    VectorXd resid = VectorXd::Zero(n);
    VectorXd w = y;
    for (int _ = 0; _ < arma[5]; ++_)
    {
      for (int l = n - 1; l > 0; --l)
      {
        w[l] -= w[l - 1];
      }
    }
    int ns = arma[4];
    for (int _ = 0; _ < arma[6]; ++_)
    {
      for (int l = n - 1; l >= ns; --l)
      {
        w[l] -= w[l - ns];
      }
    }
    for (int l = ncond; l < n; ++l)
    {
      double tmp = w[l];
      for (int j = 0; j < p; ++j)
      {
        tmp -= phi[j] * w[l - j - 1];
      }
      for (int j = 0; j < std::min(l - ncond, q); ++j)
      {
        if (l - j - 1 < 0)
        {
          continue;
        }
        tmp -= theta[j] * resid[l - j - 1];
      }
      resid[l] = tmp;
      if (!std::isnan(tmp))
      {
        nu++;
        ssq += tmp * tmp;
      }
    }
    return {ssq / nu, resid};
  }

  std::tuple<double, double, int> arima_like(CRef<VectorXd> y, CRef<VectorXd> phi,
                                             CRef<VectorXd> theta,
                                             CRef<VectorXd> delta, Ref<VectorXd> a,
                                             Ref<VectorXd> P, Ref<VectorXd> Pnew, int up,
                                             bool use_resid, Ref<VectorXd> rsResid)
  {
    int n = static_cast<int>(y.size());
    int d = static_cast<int>(delta.size());
    int rd = static_cast<int>(a.size());
    int p = static_cast<int>(phi.size());
    int q = static_cast<int>(theta.size());
    double ssq = 0.0;
    double sumlog = 0.0;
    int nu = 0;
    int r = rd - d;

    std::vector<double> anew(rd);
    std::vector<double> M(rd);
    std::vector<double> mm;
    if (d > 0)
    {
      mm.resize(rd * rd);
    }
    double tmp;
    for (int l = 0; l < n; ++l)
    {
      for (int i = 0; i < r; ++i)
      {
        if (i < r - 1)
        {
          tmp = a[i + 1];
        }
        else
        {
          tmp = 0.0;
        }
        if (i < p)
        {
          tmp += phi[i] * a[0];
        }
        anew[i] = tmp;
      }
      if (d > 0)
      {
        for (int i = r + 1; i < rd; ++i)
        {
          anew[i] = a[i - 1];
        }
        tmp = a[0];
        for (int i = 0; i < d; ++i)
        {
          tmp += delta[i] * a[r + i];
        }
        anew[r] = tmp;
      }
      if (l > up)
      {
        if (d == 0)
        {
          for (int i = 0; i < r; ++i)
          {
            double vi = 0.0;
            if (i == 0)
            {
              vi = 1.0;
            }
            else if (i - 1 < q)
            {
              vi = theta[i - 1];
            }
            for (int j = 0; j < r; ++j)
            {
              tmp = 0.0;
              if (j == 0)
              {
                tmp = vi;
              }
              else if (j - 1 < q)
              {
                tmp = vi * theta[j - 1];
              }
              if (i < p && j < p)
              {
                tmp += phi[i] * phi[j] * P[0];
              }
              if (i < r - 1 && j < r - 1)
              {
                tmp += P[i + 1 + r * (j + 1)];
              }
              if (i < p && j < r - 1)
              {
                tmp += phi[i] * P[j + 1];
              }
              if (j < p && i < r - 1)
              {
                tmp += phi[j] * P[i + 1];
              }
              Pnew[i + r * j] = tmp;
            }
          }
        }
        else
        {
          for (int i = 0; i < r; ++i)
          {
            for (int j = 0; j < rd; ++j)
            {
              tmp = 0.0;
              if (i < p)
              {
                tmp += phi[i] * P[rd * j];
              }
              if (i < r - 1)
              {
                tmp += P[i + 1 + rd * j];
              }
              mm[i + rd * j] = tmp;
            }
          }
          for (int j = 0; j < rd; ++j)
          {
            tmp = P[rd * j];
            for (int k = 0; k < d; ++k)
            {
              tmp += delta[k] * P[r + k + rd * j];
            }
            mm[r + rd * j] = tmp;
          }
          for (int i = 1; i < d; ++i)
          {
            for (int j = 0; j < rd; ++j)
            {
              mm[r + i + rd * j] = P[r + i - 1 + rd * j];
            }
          }
          for (int i = 0; i < r; ++i)
          {
            for (int j = 0; j < rd; ++j)
            {
              tmp = 0.0;
              if (i < p)
              {
                tmp += phi[i] * mm[j];
              }
              if (i < r - 1)
              {
                tmp += mm[rd * (i + 1) + j];
              }
              Pnew[j + rd * i] = tmp;
            }
          }
          for (int j = 0; j < rd; ++j)
          {
            tmp = mm[j];
            for (int k = 0; k < d; ++k)
            {
              tmp += delta[k] * mm[rd * (r + k) + j];
            }
            Pnew[rd * r + j] = tmp;
          }
          for (int i = 1; i < d; ++i)
          {
            for (int j = 0; j < rd; ++j)
            {
              Pnew[rd * (r + i) + j] = mm[rd * (r + i - 1) + j];
            }
          }
          for (int i = 0; i < q + 1; ++i)
          {
            double vi;
            if (i == 0)
            {
              vi = 1.0;
            }
            else
            {
              vi = theta[i - 1];
            }
            for (int j = 0; j < q + 1; ++j)
            {
              if (j == 0)
              {
                Pnew[i + rd * j] += vi;
              }
              else
              {
                Pnew[i + rd * j] += vi * theta[j - 1];
              }
            }
          }
        }
      }
      if (!std::isnan(y[l]))
      {
        double resid = y[l] - anew[0];
        for (int i = 0; i < d; ++i)
        {
          resid -= delta[i] * anew[r + i];
        }
        for (int i = 0; i < rd; ++i)
        {
          tmp = Pnew[i];
          for (int j = 0; j < d; ++j)
          {
            tmp += Pnew[i + (r + j) * rd] * delta[j];
          }
          M[i] = tmp;
        }
        double gain = M[0];
        for (int j = 0; j < d; ++j)
        {
          gain += delta[j] * M[r + j];
        }
        if (gain < 1e4)
        {
          nu++;
          if (gain == 0)
          {
            ssq = std::numeric_limits<double>::infinity();
          }
          else
          {
            ssq += resid * resid / gain;
          }
          sumlog += std::log(gain);
        }
        if (use_resid)
        {
          if (gain == 0)
          {
            rsResid[l] = std::numeric_limits<double>::infinity();
          }
          else
          {
            rsResid[l] = resid / std::sqrt(gain);
          }
        }
        if (gain == 0)
        {
          for (int i = 0; i < rd; ++i)
          {
            a[i] = std::numeric_limits<double>::infinity();
            for (int j = 0; j < rd; ++j)
            {
              Pnew[i + j * rd] = std::numeric_limits<double>::infinity();
            }
          }
        }
        else
        {
          for (int i = 0; i < rd; ++i)
          {
            a[i] = anew[i] + M[i] * resid / gain;
            for (int j = 0; j < rd; ++j)
            {
              P[i + j * rd] = Pnew[i + j * rd] - M[i] * M[j] / gain;
            }
          }
        }
      }
      else
      {
        std::copy(anew.begin(), anew.end(), a.data());
        std::copy(Pnew.begin(), Pnew.begin() + rd * rd, P.begin());
      }
    }
    return {ssq, sumlog, nu};
  }

  void inclu2(int np, const double *xnext, double *xrow, double ynext, double *d,
              double *rbar, double *thetab)
  {
    std::copy(xnext, xnext + np, xrow);
    int ithisr = 0;
    for (int i = 0; i < np; ++i)
    {
      if (xrow[i] != 0.0)
      {
        double xi = xrow[i];
        double di = d[i];
        double dpi = di + xi * xi;
        d[i] = dpi;
        double cbar, sbar;
        if (dpi == 0)
        {
          cbar = std::numeric_limits<double>::infinity();
          sbar = std::numeric_limits<double>::infinity();
        }
        else
        {
          cbar = di / dpi;
          sbar = xi / dpi;
        }
        for (int k = i + 1; k < np; ++k)
        {
          double xk = xrow[k];
          double rbthis = rbar[ithisr];
          xrow[k] = xk - xi * rbthis;
          rbar[ithisr++] = cbar * rbthis + sbar * xk;
        }
        double xk = ynext;
        ynext = xk - xi * thetab[i];
        thetab[i] = cbar * thetab[i] + sbar * xk;
        if (di == 0.0)
        {
          return;
        }
      }
      else
      {
        ithisr += np - i - 1;
      }
    }
  }

  void getQ0(CRef<VectorXd> phi, CRef<VectorXd> theta, Ref<VectorXd> res)
  {
    int p = static_cast<int>(phi.size());
    int q = static_cast<int>(theta.size());
    int r = std::max(p, q + 1);
    int np = r * (r + 1) / 2;
    int nrbar = np * (np - 1) / 2;
    int ind = 0;

    std::vector<double> V(np);
    for (int j = 0; j < r; ++j)
    {
      double vj = 0.0;
      if (j == 0)
      {
        vj = 1.0;
      }
      else if (j - 1 < q)
      {
        vj = theta[j - 1];
      }
      for (int i = j; i < r; ++i)
      {
        double vi = 0.0;
        if (i == 0)
        {
          vi = 1.0;
        }
        else if (i - 1 < q)
        {
          vi = theta[i - 1];
        }
        V[ind++] = vi * vj;
      }
    }
    if (r == 1)
    {
      if (p == 0)
      {
        res[0] = 1.0;
      }
      else
      {
        res[0] = 1.0 / (1 - phi[0] * phi[0]);
      }
      return;
    }
    if (p > 0)
    {
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
      for (int j = 0; j < r; ++j)
      {
        double phij = j < p ? phi[j] : 0.0;
        xnext[indj++] = 0.0;
        int indi = npr1 + j;
        for (int i = j; i < r; ++i)
        {
          double ynext = V[ind++];
          double phii = i < p ? phi[i] : 0.0;
          if (j != r - 1)
          {
            xnext[indj] = -phii;
            if (i != r - 1)
            {
              xnext[indi] -= phij;
              xnext[++ind1] = -1.0;
            }
          }
          xnext[npr] = -phii * phij;
          if (++ind2 >= np)
          {
            ind2 = 0;
          }
          xnext[ind2] += 1.0;
          inclu2(np, xnext.data(), xrow.data(), ynext, res.data(), rbar.data(),
                 thetab.data());
          xnext[ind2] = 0.0;
          if (i != r - 1)
          {
            xnext[indi++] = 0.0;
            xnext[ind1] = 0.0;
          }
        }
      }
      int ithisr = nrbar - 1;
      int im = np - 1;
      for (int i = 0; i < np; ++i)
      {
        double bi = thetab[im];
        int jm = np - 1;
        for (int j = 0; j < i; ++j)
        {
          bi -= rbar[ithisr--] * res[jm--];
        }
        res[im--] = bi;
      }
      ind = npr;
      for (int i = 0; i < r; ++i)
      {
        xnext[i] = res[ind++];
      }
      ind = np - 1;
      ind1 = npr - 1;
      for (int i = 0; i < npr; ++i)
      {
        res[ind--] = res[ind1--];
      }
      std::copy(xnext.begin(), xnext.begin() + r, res.data());
    }
    else
    {
      int indn = np;
      ind = np;
      for (int i = 0; i < r; ++i)
      {
        for (int j = 0; j < i + 1; ++j)
        {
          --ind;
          res[ind] = V[ind];
          if (j != 0)
          {
            res[ind] += res[--indn];
          }
        }
      }
    }
    ind = np;
    for (int i = r - 1; i > 0; --i)
    {
      for (int j = r - 1; j > i - 1; --j)
      {
        res[r * i + j] = res[--ind];
      }
    }
    for (int i = 0; i < r - 1; ++i)
    {
      for (int j = i + 1; j < r; ++j)
      {
        res[i + r * j] = res[j + r * i];
      }
    }
  }

  RowMatrixXd arima_gradtrans(CRef<VectorXd> x, CRef<VectorXi> arma)
  {
    double eps = 1e-3;
    int n = static_cast<int>(x.size());
    int mp = arma[0];
    int mq = arma[1];
    int msp = arma[2];

    auto w1 = std::array<double, 100>();
    auto w2 = std::array<double, 100>();
    auto w3 = std::array<double, 100>();
    RowMatrixXd out = RowMatrixXd::Identity(n, n);
    if (mp > 0)
    {
      std::copy(x.data(), x.data() + mp, w1.begin());
      partrans(mp, w1.data(), w2.data());
      for (int i = 0; i < mp; ++i)
      {
        w1[i] += eps;
        partrans(mp, w1.data(), w3.data());
        for (int j = 0; j < mp; ++j)
        {
          out(i, j) = (w3[j] - w2[j]) / eps;
        }
        w1[i] -= eps;
      }
    }
    if (msp > 0)
    {
      int v = mp + mq;
      std::copy(x.data() + v, x.data() + v + msp, w1.begin());
      partrans(msp, w1.data(), w2.data());
      for (int i = 0; i < msp; ++i)
      {
        w1[i] += eps;
        partrans(msp, w1.data(), w3.data());
        for (int j = 0; j < msp; ++j)
        {
          out(i + v, j + v) = (w3[j] - w2[j]) / eps;
        }
        w1[1] -= eps;
      }
    }
    return out;
  }

  VectorXd arima_undopars(CRef<VectorXd> x, CRef<VectorXi> arma)
  {
    int mp = arma[0];
    int mq = arma[1];
    int msp = arma[2];
    VectorXd out = x;
    if (mp > 0)
    {
      partrans(mp, x.data(), out.data());
    }
    int v = mp + mq;
    if (msp > 0)
    {
      partrans(msp, x.data() + v, out.data() + v);
    }
    return out;
  }

  void invpartrans(int p, CRef<VectorXd> phi, Ref<VectorXd> out)
  {
    std::copy(phi.begin(), phi.begin() + p, out.begin());
    std::vector<double> work(phi.begin(), phi.begin() + p);
    for (int j = p - 1; j > 0; --j)
    {
      double a = out[j];
      for (int k = 0; k < j; ++k)
      {
        work[k] = (out[k] + a * out[j - k - 1]) / (1 - a * a);
      }
      std::copy(work.begin(), work.begin() + j, out.begin());
    }
    for (int j = 0; j < p; ++j)
    {
      out[j] = std::atanh(out[j]);
    }
  }

  void init(py::module_ &m)
  {
    py::module_ arima = m.def_submodule("arima");
    arima.def("arima_css", &arima_css);
    arima.def("arima_like", &arima_like);
    arima.def("getQ0", &getQ0);
    arima.def("arima_gradtrans", &arima_gradtrans);
    arima.def("arima_undopars", &arima_undopars);
    arima.def("invpartrans", &invpartrans);
    arima.def("arima_transpar", &arima_transpar);
  }
}
