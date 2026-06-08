cmake_minimum_required(VERSION 3.20)
project(ckks_mnist LANGUAGES CXX CUDA)

# RTX 4060 = sm_89; 换 GPU 时改这里或传 -DCMAKE_CUDA_ARCHITECTURES=...
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif ()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# 引用同级的 phantom-fhe 源码(与 ckks-demo 相同方式), 关闭其 examples/bench/test。
set(PHANTOM_DIR "${CMAKE_SOURCE_DIR}/../phantom-fhe" CACHE PATH "Phantom-FHE source dir")

set(PHANTOM_ENABLE_EXAMPLE OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_BENCH   OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_TEST    OFF CACHE BOOL "" FORCE)
set(PHANTOM_ENABLE_PYTHON_BINDING OFF CACHE BOOL "" FORCE)

include_directories(${PHANTOM_DIR}/include)
add_subdirectory(${PHANTOM_DIR} ${CMAKE_BINARY_DIR}/phantom-build)

add_executable(ckks_mnist ckks_mnist.cu)
target_include_directories(ckks_mnist PRIVATE ${PHANTOM_DIR}/include)
target_link_libraries(ckks_mnist PRIVATE Phantom)
set_target_properties(ckks_mnist PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
