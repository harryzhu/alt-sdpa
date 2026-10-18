cmake_minimum_required(VERSION 3.27)
project(alt_sdpa_ext VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -frtti")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


option(BUILD_SHARED_LIBS "Build extensions as a shared library" ON)


find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE 
  OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
  RESULT_VARIABLE NANOBIND_CMAKE_DIR_RESULT
  )
if(NOT NANOBIND_CMAKE_DIR_RESULT EQUAL 0)
    message(FATAL_ERROR "Failed to get nanobind cmake dir. Is nanobind installed?")
endif()
set(CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR};${CMAKE_PREFIX_PATH}")
find_package(nanobind CONFIG REQUIRED PATHS "${NANOBIND_CMAKE_DIR}")

include_directories("/opt/anaconda3/lib/python3.13/site-packages/mlx/include")

nanobind_add_module(alt_sdpa_ext src/alt_sdpa.cpp)

# 自动链接BLAS库
if(APPLE)
    find_library(ACCELERATE_LIB Accelerate REQUIRED)
    # 加这两行
    target_compile_definitions(alt_sdpa_ext PRIVATE
        ACCELERATE_NEW_LAPACK
        ACCELERATE_LAPACK_ILP64
    )
    target_link_libraries(alt_sdpa_ext PRIVATE ${ACCELERATE_LIB})
else()
    find_library(OPENBLAS_LIB openblas REQUIRED)
    target_link_libraries(alt_sdpa_ext PRIVATE ${OPENBLAS_LIB})
endif()

