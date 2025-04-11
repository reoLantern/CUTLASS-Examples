cmake_minimum_required(VERSION 3.22)

project(CUTE-WGMMA-TMA-SM90 VERSION 0.0.1 LANGUAGES CXX CUDA)

# Set the CUDA architecture to compile the code for
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
set(CUTE_WGMMA_TMA_SM90 cute_wgmma_tma_sm90)
add_executable(
    ${CUTE_WGMMA_TMA_SM90}
    cute_wgmma_tma_sm90.cu
)
target_include_directories(${CUTE_WGMMA_TMA_SM90} PRIVATE ${CUTLASS_INCLUDE_DIR} ${CUTLASS_UTILS_INCLUDE_DIR})
target_compile_definitions(${CUTE_WGMMA_TMA_SM90} PRIVATE CUTLASS_ARCH_MMA_SM90A_SUPPORTED)
set_target_properties(${CUTE_WGMMA_TMA_SM90} PROPERTIES CUDA_ARCHITECTURES 90a)
target_compile_options(${CUTE_WGMMA_TMA_SM90} PRIVATE --expt-relaxed-constexpr)
target_link_libraries(cute_wgmma_tma_sm90 PRIVATE cuda dl)
