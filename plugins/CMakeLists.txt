# Builds libtrt_warp.so on a machine with CUDA and TensorRT 10 (e.g. JetPack 6 on a Jetson).
#   cmake -S plugins -B build/plugins -DTENSORRT_ROOT=/usr && cmake --build build/plugins
cmake_minimum_required(VERSION 3.22)
project(trt_warp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(TENSORRT_ROOT "/usr" CACHE PATH "TensorRT install prefix")

set_source_files_properties(trt_warp.cpp PROPERTIES LANGUAGE CUDA)
add_library(trt_warp SHARED trt_warp.cpp)
target_include_directories(trt_warp PRIVATE ${TENSORRT_ROOT}/include ${TENSORRT_ROOT}/include/aarch64-linux-gnu
                                            ${TENSORRT_ROOT}/include/x86_64-linux-gnu)
find_library(NVINFER nvinfer HINTS ${TENSORRT_ROOT}/lib ${TENSORRT_ROOT}/lib/aarch64-linux-gnu ${TENSORRT_ROOT}/lib/x86_64-linux-gnu)
target_link_libraries(trt_warp PRIVATE ${NVINFER})
set_target_properties(trt_warp PROPERTIES CUDA_ARCHITECTURES "72;87")
