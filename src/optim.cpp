#include <pybind11/pybind11.h>

#include "optim.h"

namespace optim {
namespace py = pybind11;
void init(py::module_ &m) {
  py::module_ optim = m.def_submodule("optim");
  py::class_<optim::Result>(optim, "OptimResult")
      .def(py::init<>())
      .def_readonly("x", &optim::Result::x)
      .def_readonly("fun", &optim::Result::fun)
      .def_readonly("nit", &optim::Result::nit);
}
} // namespace optim
