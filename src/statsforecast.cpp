#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace ets {
void init(py::module_ &);
}

namespace arima {
void init(py::module_ &);
}

namespace theta {
void init(py::module_ &);
}

namespace ses {
void init(py::module_ &);
}

namespace garch {
void init(py::module_ &);
}

namespace ces {
void init(py::module_ &);
}

namespace tbats_ns {
void init(py::module_ &);
}

namespace mfles {
void init(py::module_ &);
}

PYBIND11_MODULE(_lib, m) {
  arima::init(m);
  ets::init(m);
  theta::init(m);
  ses::init(m);
  garch::init(m);
  ces::init(m);
  tbats_ns::init(m);
  mfles::init(m);
}
