cmake_minimum_required(VERSION 3.15...3.25)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(pybind11 CONFIG REQUIRED)

include_directories(
  include/statsforecast
  external_libs/eigen)
file(GLOB SOURCES CONFIGURE_DEPENDS src/*.cpp)
pybind11_add_module(_lib ${SOURCES})

install(TARGETS _lib DESTINATION .)
