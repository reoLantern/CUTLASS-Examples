cmake_minimum_required(VERSION 3.28)

project(CUTE-TENSORMAP-VIEW VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_TENSORMAP_VIEW cute_tensormap_view)
add_executable(
    ${CUTE_TENSORMAP_VIEW}
    cute_tensormap_view.cu
)
target_include_directories(${CUTE_TENSORMAP_VIEW} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_TENSORMAP_VIEW} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_TENSORMAP_VIEW} PRIVATE --expt-relaxed-constexpr)
target_link_libraries(cute_tensormap_view PRIVATE cuda dl)
