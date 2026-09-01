#include <pybind11/pybind11.h>
#include "distributions.h"

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

  // Tail bounds from distributions.h, read by statsforecast/distributions.py.
  py::module_ distributions = m.def_submodule("distributions");
  // Enum is already registered on ets (same C++ type); alias it as ces/theta do.
  distributions.attr("Distribution") = m.attr("ets").attr("Distribution");
  distributions.def(
      "tail_bounds",
      [](dist::Distribution d) {
        const dist::TailBounds b = dist::tail_bounds(d);
        return py::make_tuple(py::make_tuple(b.scale_lo, b.shape_lo),
                              py::make_tuple(b.scale_hi, b.shape_hi));
      },
      py::arg("distribution"),
      "(lower, upper) for the [log_scale, shape] tail; unbounded for "
      "Normal/Laplace.");
}
