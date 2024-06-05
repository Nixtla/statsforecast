#pragma once

extern "C" {
double arima_css(const double *y, int n, const int *arma, const double *phi,
                 int p, const double *theta, int q);
double arma_css_op(const double *p, const double *y, int n, const double *coef,
                   const int *arma, const bool *mask);
void arima_like(const double *y, int n, const double *phi, int p,
                const double *theta, int q, const double *delta, int d,
                double *a, int rd, double *P, double *Pnew, int up,
                bool use_resid, double *ssq, double *sumlog, int *nu,
                double *rsResid);
void getQ0(const double *phi, int p, const double *theta, int q, double *res);
double armafn(const double *p, const double *y, int n, const double *delta,
              int d, const double *coef, const int *arma, const bool *mask,
              bool trans, double *P, double *Pn, double *a, double *T);
void upARIMA(const double *phi, int p, const double *theta, int q, int d,
             double *Pn, double *T, double *a);
void arima_gradtrans(const double *x, int n, const int *arma, double *out);
void invpartrans(int p, const double *phi, double *out);
}
