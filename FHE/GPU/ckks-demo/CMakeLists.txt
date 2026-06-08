cmake_minimum_required(VERSION 3.20)
project(ckks_demo LANGUAGES CXX CUDA)

# RTX 4060 = sm_89??? GPU ????? native
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# ---------------------------------------------------------------------------
# ????? Phantom-FHE ???????????????????
# ?? Phantom ??? examples/bench/test????????
# ---------------------------------------------------------------------------
set(PHANTOM_DIR "${CMAKE_SOURCE_DIR}/../phantom-fhe" CACHE PATH "Phantom-FHE source dir")

set(PHANTOM_ENABLE_EXAMPLE OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_BENCH   OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_TEST    OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_PYTHON_BINDING OFF CACHE BOOL "" FORCE)

# Phantom ? src/CMakeLists.txt ?? CMAKE_SOURCE_DIR/include?
# ? add_subdirectory ?????????????????????????
# ? Phantom ?????????????????????????
include_directories(${PHANTOM_DIR}/include)

add_subdirectory(${PHANTOM_DIR} ${CMAKE_BINARY_DIR}/phantom-build)

# ---------------------------------------------------------------------------
# ????????
# ---------------------------------------------------------------------------
add_executable(ckks_demo ckks_demo.cu)
target_include_directories(ckks_demo PRIVATE ${PHANTOM_DIR}/include)
target_link_libraries(ckks_demo PRIVATE Phantom)
set_target_properties(ckks_demo PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
