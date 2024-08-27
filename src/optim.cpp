#include <pybind11/pybind11.h>

#include "optim.h"

namespace py = pybind11;

void init_optim(py::module_ &m) {
  py::module_ optim = m.def_submodule("optim");
  py::class_<OptimResult>(optim, "OptimResult")
      .def(py::init<>())
      .def_readonly("x", &OptimResult::x)
      .def_readonly("fun", &OptimResult::fun)
      .def_readonly("nit", &OptimResult::nit);
}
