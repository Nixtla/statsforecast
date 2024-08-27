#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ETS {
void init(py::module_ &);
}
void init_optim(py::module_ &);

PYBIND11_MODULE(_lib, m) {
  ETS::init(m);
  init_optim(m);
}
