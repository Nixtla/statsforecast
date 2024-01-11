cmake_minimum_required(VERSION 3.25)
project(statsforecast)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 17)

if(APPLE)
    set(CMAKE_SHARED_LIBRARY_SUFFIX ".so")
endif()

if(UNIX)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fPIC -O3 -Wall -Wextra -Wpedantic")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Ob2 /Ot /Oy /W4")
endif()

if(SKBUILD)
    set(LIBRARY_OUTPUT_PATH ${SKBUILD_PLATLIB_DIR}/statsforecast/lib)
else()
    set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/statsforecast/lib)
endif()

include_directories(include)
add_library(statsforecast SHARED src/ets.cpp)
if(MSVC)
    set_target_properties(coreforecast PROPERTIES OUTPUT_NAME "libstatsforecast")
endif()
