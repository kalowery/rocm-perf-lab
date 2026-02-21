cmake_minimum_required(VERSION 3.15)
project(rocm_perf_isolate)

# Ensure ROCm packages are discoverable
set(CMAKE_PREFIX_PATH "/opt/rocm" ${CMAKE_PREFIX_PATH})

find_package(hsa-runtime64 REQUIRED)

add_library(rocm_perf_isolate SHARED isolate_tool.cpp)

# Compatibility include shim for ROCm header layout
 target_include_directories(rocm_perf_isolate PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/rocm_compat
)

 target_link_libraries(rocm_perf_isolate
    PRIVATE hsa-runtime64::hsa-runtime64
)

set_target_properties(rocm_perf_isolate PROPERTIES
    CXX_STANDARD 17
    POSITION_INDEPENDENT_CODE ON
)
