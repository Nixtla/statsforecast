#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ets
{
  void init(py::module_ &);
}

namespace arima
{
  void init(py::module_ &);
}

PYBIND11_MODULE(_lib, m)
{
  arima::init(m);
  ets::init(m);
}
