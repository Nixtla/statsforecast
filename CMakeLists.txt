cmake_minimum_required(VERSION 3.25)
project(statsforecast)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(SKBUILD)
  set(LIBRARY_OUTPUT_PATH ${SKBUILD_PLATLIB_DIR}/statsforecast)
else()
  set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/python/statsforecast)
endif()

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
set(PYBIND11_FINDPYTHON ON)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_DIR)
find_package(pybind11 CONFIG REQUIRED)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --includes
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE pybind11_INCLUDES)

include_directories("include/statsforecast" "external_libs/eigen" ${pybind11_INCLUDES})
file(GLOB SOURCES src/*.cpp)
pybind11_add_module(_lib ${SOURCES})
install(TARGETS _lib DESTINATION .)
