/******************************************************************************
 *  本示例演示如何使用 CUTE 在 GPU 上定义 4D layout
 *  然后把该布局对应的全局内存数据拷贝到共享内存(线性)并再写回全局内存。
 ******************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include "cutlass/arch/barrier.h"

#ifndef CUTE_ARCH_TMA_SM90_ENABLED
#define CUTE_ARCH_TMA_SM90_ENABLED
#endif

#define PRINT_LAYOUT_MODE 1

constexpr int OUT_ROW = 4;
constexpr int OUT_COL = 4;
constexpr int IN_ROW  = 16;
constexpr int IN_COL  = 16;

constexpr int IN_COL_STRIDE  = 1;
constexpr int IN_ROW_STRIDE  = IN_COL;
constexpr int OUT_ROW_STRIDE = IN_ROW * IN_ROW_STRIDE;
constexpr int OUT_COL_STRIDE = OUT_ROW * OUT_ROW_STRIDE;

// constexpr int IN_COL_STRIDE  = 1;
// constexpr int OUT_COL_STRIDE = IN_COL;
// constexpr int IN_ROW_STRIDE  = OUT_COL * OUT_COL_STRIDE;
// constexpr int OUT_ROW_STRIDE = IN_ROW * IN_ROW_STRIDE;

constexpr int vector_shape = IN_ROW * IN_COL * OUT_ROW * OUT_COL;

// 检查 CUDA 错误的宏
#define CUDA_CHECK(expr)                                                                    \
    do                                                                                        \
    {                                                                                         \
        cudaError_t status = (expr);                                                          \
        if (status != cudaSuccess)                                                            \
        {                                                                                     \
            std::cerr << "CUDA Error: " << cudaGetErrorString(status)                        \
                      << " at line " << __LINE__ << std::endl;                                \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    } while (0)

/// Kernel: 将带有 4D layout 的全局输入数据复制到共享内存(线性)，再复制回一个全局输出缓冲。
template<class TmaLoadAtom>
__global__ void copy_4d_to_smem_then_back_kernel(float const *__restrict__ global_in,
                                                  float *__restrict__ global_out,
                                                  /* 传入事先构造好的 TMA 拷贝原语 */
                                                  TmaLoadAtom tma_load)
{
    printf("debug_copy kernel\n");

    using namespace cute;

    // 1) 在 device 端同样定义 4D layout（与 host 构造的一致）
    auto shape0 = make_shape(Int<IN_ROW>{}, Int<OUT_ROW>{});
    auto shape1 = make_shape(Int<IN_COL>{}, Int<OUT_COL>{});
    auto big_shape = make_shape(shape0, shape1);

    auto stride0 = make_stride(Int<IN_ROW_STRIDE>{}, Int<OUT_ROW_STRIDE>{});
    auto stride1 = make_stride(Int<IN_COL_STRIDE>{}, Int<OUT_COL_STRIDE>{});
    auto big_stride = make_stride(stride0, stride1);

    auto gmem_layout = make_layout(big_shape, big_stride);

    // 2) 构造全局 tensor
    Tensor gA_in  = make_tensor(make_gmem_ptr(global_in), gmem_layout);
    Tensor gA_out = make_tensor(global_out, gmem_layout);

    // 3) 声明共享内存区（必须用 extern __shared__）
    extern __shared__ __align__(16) unsigned char smem_buffer[];
    float *smem_ptr = reinterpret_cast<float *>(smem_buffer);
    Tensor sA = make_tensor(make_smem_ptr(smem_ptr), gmem_layout);

    // 4) 得到扁平化 view（group_modes ），与 host 上构造时一致
    auto gA_in_flat = group_modes<0,2>(gA_in);
    auto sA_flat    = group_modes<0,2>(sA);
    auto flat_shape = shape(gA_in_flat);

    // ---------------------------
    // 5) 准备 TMA 拷贝原语所需 barrier
    // ---------------------------
    using cute::SM90_TMA_LOAD;
    using ProducerBarrier = cutlass::arch::ClusterTransactionBarrier;

    __shared__ alignas(16) uint64_t tma_load_barrier[1];

    int warp_idx  = threadIdx.x >> 5;
    bool is_warp0 = (warp_idx == 0);
    __syncthreads();

    if (threadIdx.x == 0)
    {
        ProducerBarrier::init(&tma_load_barrier[0], 1);   // 需等待1次 arrive
    }
    __syncthreads();

    if (is_warp0)
    {
        size_t bytes = size_t(size(gA_in_flat)) * sizeof(float);
        ProducerBarrier::arrive_and_expect_tx(&tma_load_barrier[0], bytes);
    }

    // ---------------------------
    // 6) 使用传入的 tma_load 进行 TMA load 操作
    // ---------------------------
    copy(tma_load.with(tma_load_barrier[0]), tma_load.get_tma_tensor(flat_shape), sA_flat);

    if (is_warp0)
    {
        ProducerBarrier::wait(&tma_load_barrier[0], 0);
    }

    __syncthreads();

    // ---------------------------
    // 7) 将 shared memory 数据拷回到 global_out（线性遍历）
    // ---------------------------
    int count = 0; // 线性遍历共享内存
    for (int outer_col = 0; outer_col < OUT_COL; ++outer_col)
    {
        for (int outer_row = 0; outer_row < OUT_ROW; ++outer_row)
        {
            for (int inner_row = 0; inner_row < IN_ROW; ++inner_row)
            {
                for (int inner_col = 0; inner_col < IN_COL; ++inner_col)
                {
                    auto row_coord = make_coord(inner_row, outer_row);
                    auto col_coord = make_coord(inner_col, outer_col);
                    auto coord     = make_coord(row_coord, col_coord);

                    int gmem_offset = gmem_layout(coord);
                    // printf("gmem_offset %d\n", gmem_offset);
                    gA_out(gmem_offset) = sA(count);
                    ++count;
                }
            }
        }
    }
}


// -------------------- print_data 函数，模仿 cute::print_layout() -------------------
template <class Layout> CUTE_HOST_DEVICE void print_data(Layout const &layout, float const *data)
{
    CUTE_STATIC_ASSERT_V(rank(layout) == cute::Int<2>{});

    int idx_width = num_digits(cosize(layout)) + 2;
    const char *delim = "+-----------------------";

    printf("\nData:\n");

    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        printf("  %*d ", idx_width - 2, n);
    }
    printf("\n");

    for (int m = 0; m < cute::size<0>(layout); ++m)
    {
        printf("    ");
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            printf("%.*s", idx_width + 1, delim);
        }
        printf("+\n");

        printf("%2d  ", m);
        for (int n = 0; n < cute::size<1>(layout); ++n)
        {
            int offset = layout(m, n);
            printf("| %*g ", idx_width - 2, data[offset]);
        }
        printf("|\n");
    }
    printf("    ");
    for (int n = 0; n < cute::size<1>(layout); ++n)
    {
        printf("%.*s", idx_width + 1, delim);
    }
    printf("+\n");
}


//---------------------------------------------------------------------
// main() 函数：
// 1. 在 host 上构造 4D layout，并用该 layout 打印数据；
// 2. 分配并初始化 host/device 内存；
// 3. 在 host 上利用全局（device）指针和 layout 构造 tensor，
//    进而计算扁平化 view 及 shape，从而构造 TMA 拷贝原语 tma_load；
// 4. 将 tma_load 作为 kernel 参数传入，完成 global->shared->global 的拷贝；
// 5. 拷回 host 检查结果。
//---------------------------------------------------------------------
int main()
{
    using namespace cute;

    // --- 1) 在 Host 上构造相同 layout 并打印 ---
    auto shape0 = make_shape(Int<IN_ROW>{}, Int<OUT_ROW>{});
    auto shape1 = make_shape(Int<IN_COL>{}, Int<OUT_COL>{});
    auto big_shape = make_shape(shape0, shape1);

    auto stride0 = make_stride(Int<IN_ROW_STRIDE>{}, Int<OUT_ROW_STRIDE>{});
    auto stride1 = make_stride(Int<IN_COL_STRIDE>{}, Int<OUT_COL_STRIDE>{});
    auto big_stride = make_stride(stride0, stride1);

    auto layout_4d = make_layout(big_shape, big_stride);

    std::cout << "\n[Host] 4D layout description:\n";
    if (PRINT_LAYOUT_MODE)
        cute::print_layout(layout_4d);
    std::cout << std::endl;

    size_t total_elems = size(layout_4d);

    // --- 2) 分配并初始化 host 的 input/output 数组 ---
    std::vector<float> h_in(total_elems), h_out(total_elems, -1.0f);
    for (size_t i = 0; i < total_elems; ++i) h_in[i] = static_cast<float>(i);

    // --- 3) 分配 device 内存，并将 host 数据拷贝到 device ---
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_in, total_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_out, total_elems * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elems * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out, h_out.data(), total_elems * sizeof(float), cudaMemcpyHostToDevice));

    // --- 4) 在 host 上构造 TMA 拷贝原语 ---
    // 用全局内存指针 d_in 构造全局 tensor，并计算扁平化 view 和 shape
    Tensor gA_in_host = make_tensor(make_gmem_ptr(d_in), layout_4d);
    auto gA_in_flat  = group_modes<0,2>(gA_in_host);
    auto flat_shape  = shape(gA_in_flat);

    // 为共享内存构造一个 dummy tensor（仅用来获得 layout 信息）
    Tensor sA_host = make_tensor(make_smem_ptr((float*)nullptr), layout_4d);
    auto sA_flat_host = group_modes<0,2>(sA_host);

    // 这里调用的是 host 版本的 make_tma_atom，可正常构造 TMA 拷贝原语
    cute::Copy_Atom tma_load = make_tma_atom(cute::SM90_TMA_LOAD{}, gA_in_flat, layout(sA_flat_host), flat_shape);

    // --- 5) 启动 kernel：传入 tma_load，同时分配足够共享内存 ---
    dim3 gridDim(1);
    dim3 blockDim(1);
    size_t smem_bytes = total_elems * sizeof(float);

    copy_4d_to_smem_then_back_kernel<<<gridDim, blockDim, smem_bytes>>>(d_in, d_out, tma_load);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 6) 将结果拷贝回 host 并打印 ---
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, total_elems * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n[Host] Print complete input data as table using layout_4d:\n";
    if (PRINT_LAYOUT_MODE)
        print_data(layout_4d, h_in.data());
    std::cout << "\n[Host] Print complete output data as table using layout_4d:\n";
    if (PRINT_LAYOUT_MODE)
        print_data(layout_4d, h_out.data());

    // --- 7) 检查结果 ---
    int error_count = 0;
    for (size_t i = 0; i < total_elems; ++i)
    {
        if (h_in[i] != h_out[i])
        {
            ++error_count;
            if (error_count < 10)
            {
                std::cerr << "Mismatch at index " << i 
                          << ": input=" << h_in[i] 
                          << ", output=" << h_out[i] << "\n";
            }
        }
    }
    if (error_count == 0)
    {
        std::cout << "Success! All " << total_elems << " elements match after shared-memory copy.\n";
    }
    else
    {
        std::cerr << "Found " << error_count << " mismatches.\n";
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
