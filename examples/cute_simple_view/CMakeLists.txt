cmake_minimum_required(VERSION 3.28)

project(CUTE-SIMPLE-VIEW VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_SIMPLE_VIEW cute_simple_view)
add_executable(
    ${CUTE_SIMPLE_VIEW}
    cute_simple_view.cu
)
target_include_directories(${CUTE_SIMPLE_VIEW} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
set_target_properties(${CUTE_SIMPLE_VIEW} PROPERTIES CUDA_ARCHITECTURES native)
target_compile_options(${CUTE_SIMPLE_VIEW} PRIVATE --expt-relaxed-constexpr)
