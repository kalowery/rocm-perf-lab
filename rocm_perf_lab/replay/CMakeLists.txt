cmake_minimum_required(VERSION 3.15)
project(rocm_perf_replay)

# Ensure ROCm packages are discoverable
set(CMAKE_PREFIX_PATH "/opt/rocm" ${CMAKE_PREFIX_PATH})

find_package(hsa-runtime64 REQUIRED)

add_executable(rocm_perf_replay replay.cpp)

target_link_libraries(rocm_perf_replay
    PRIVATE hsa-runtime64::hsa-runtime64
)

set_target_properties(rocm_perf_replay PROPERTIES
    CXX_STANDARD 17
)
