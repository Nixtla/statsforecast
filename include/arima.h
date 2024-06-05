#pragma once

extern "C" {
double arima_css(const double *y, int n, const int *arma, const double *phi,
                 int p, const double *theta, int q, double *resid);
void arima_like(const double *y, int n, const double *phi, int p,
                const double *theta, int q, const double *delta, int d,
                double *a, int rd, double *P, double *Pnew, int up,
                bool use_resid, double *ssq, double *sumlog, int *nu,
                double *rsResid);
void getQ0(const double *phi, int p, const double *theta, int q, double *res);
void arima_gradtrans(const double *x, int n, const int *arma, double *out);
void arima_undopars(const double *x, const int *arma, double *out);
void invpartrans(int p, const double *phi, double *out);
void arima_transpar(const double *params_in, const int *arma, bool trans,
                    double *phi, double *theta);
}
