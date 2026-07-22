cmake_minimum_required(VERSION 3.15...3.25)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)

set(PYBIND11_FINDPYTHON ON)

if(DEFINED ENV{USE_PREBUILT_EXT})
    # Skip compiling entirely — copy the extension already installed
    # from the PyPI wheel instead.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c
            "import ${SKBUILD_PROJECT_NAME}, os; print(os.path.dirname(${SKBUILD_PROJECT_NAME}.__file__))"
        OUTPUT_VARIABLE PREBUILT_PKG_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PREBUILT_LOOKUP_RESULT
    )

    if(NOT PREBUILT_LOOKUP_RESULT EQUAL 0 OR PREBUILT_PKG_DIR STREQUAL "")
        message(FATAL_ERROR
            "USE_PREBUILT_EXT is set but could not locate an installed "
	    "${SKBUILD_PROJECT_NAME} package. Please install the package from PyPI first"
            " so the compiled extension is available to copy.")
    endif()

    file(GLOB PREBUILT_EXT "${PREBUILT_PKG_DIR}/_lib*")
    list(LENGTH PREBUILT_EXT PREBUILT_EXT_COUNT)
    if(PREBUILT_EXT_COUNT EQUAL 0)
        message(FATAL_ERROR
            "USE_PREBUILT_EXT is set but no _lib* extension was found in "
            "${PREBUILT_PKG_DIR}. Check that the PyPI install succeeded.")
    endif()

    message(STATUS "Using prebuilt extension: ${PREBUILT_EXT}")
    install(FILES ${PREBUILT_EXT} DESTINATION .)
else()
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    find_package(pybind11 CONFIG REQUIRED)

    include_directories(
      include/statsforecast
      external_libs/eigen)
    file(GLOB SOURCES src/*.cpp)
    pybind11_add_module(_lib ${SOURCES})

    install(TARGETS _lib DESTINATION .)
endif()
